# %% [markdown]
# # Official Bluesky moderation labels pipeline
#
# Notebook-style script that:
# 1. Reads the DID universe from `datasets/fingerprints.parquet` with DuckDB.
# 2. Queries the public `com.atproto.label.queryLabels` endpoint in resumable DID batches.
# 3. Stores an event-level long table incrementally.
# 4. Builds account-level moderation features with pandas.
# 5. Joins the features back to the original parquet with DuckDB.

# %%
from __future__ import annotations

import ast
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import pandas as pd


# %%
# Configuration. The batch size is intentionally easy to change.
OFFICIAL_LABELER_DID = "did:plc:ar7c4by46qjdydhdevvrndac"
PUBLIC_XRPC_BASE_URL = "https://public.api.bsky.app"
QUERY_LABELS_NSID = "com.atproto.label.queryLabels"

BATCH_SIZE = 100
API_LIMIT = 250
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 32.0
REQUEST_TIMEOUT_SECONDS = 60
CHECKPOINT_VERSION = 1

LONG_LABEL_COLUMNS = ["did", "uri", "val", "cts", "src", "neg"]


# %%
def resolve_project_root() -> Path:
    candidates: list[Path] = []
    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        candidates.extend([script_path.parent, script_path.parent.parent])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    for candidate in candidates:
        if (candidate / "datasets" / "model" / "fingerprints.parquet").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate

    return cwd


PROJECT_ROOT = resolve_project_root()
INPUT_PARQUET_PATH = PROJECT_ROOT / "datasets" / "model" / "fingerprints.parquet"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "model" / "labeller"
LONG_CHUNK_DIR = OUTPUT_DIR / "official_labels_long_batches"
CHECKPOINT_PATH = OUTPUT_DIR / "official_labels_checkpoint.json"
FINAL_LONG_PATH = OUTPUT_DIR / "official_labels_long.parquet"
ACCOUNT_LEVEL_PATH = OUTPUT_DIR / "official_labels_account_level.parquet"
ENRICHED_PATH = OUTPUT_DIR / "fingerprints_enriched_with_official_labels.parquet"


# %%
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_sql(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def normalize_list_cell(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        if isinstance(converted, tuple):
            return list(converted)
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    return []


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LONG_CHUNK_DIR.mkdir(parents=True, exist_ok=True)


def parse_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {}
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_checkpoint(
    *,
    did_column_name: str,
    total_batches: int,
    last_completed_batch_index: int,
    total_labels_collected_so_far: int,
    total_unique_dids: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "input_parquet": str(INPUT_PARQUET_PATH),
        "did_column_name": did_column_name,
        "batch_size": BATCH_SIZE,
        "total_batches": total_batches,
        "last_completed_batch_index": last_completed_batch_index,
        "total_labels_collected_so_far": total_labels_collected_so_far,
        "total_unique_dids": total_unique_dids,
        "last_successful_write": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_tuple_like_column_name(column_name: str) -> tuple[Any, ...] | None:
    try:
        parsed = ast.literal_eval(column_name)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, tuple):
        return parsed
    return None


def is_did_column_name(column_name: str) -> bool:
    if column_name == "did":
        return True
    parsed = parse_tuple_like_column_name(column_name)
    if parsed and len(parsed) >= 1:
        return str(parsed[0]) == "did"
    return False


def discover_did_column_name(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> str:
    schema_df = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path_sql(parquet_path)}')"
    ).fetchdf()
    for column_name in schema_df["column_name"].tolist():
        if is_did_column_name(str(column_name)):
            return str(column_name)
    available = ", ".join(map(str, schema_df["column_name"].tolist()))
    raise ValueError(f"Could not find a DID column in {parquet_path}. Columns: {available}")


def extract_unique_dids(
    con: duckdb.DuckDBPyConnection, parquet_path: Path, did_column_name: str
) -> list[str]:
    did_sql = quote_identifier(did_column_name)
    query = f"""
        SELECT DISTINCT CAST({did_sql} AS VARCHAR) AS did
        FROM read_parquet('{path_sql(parquet_path)}')
        WHERE {did_sql} IS NOT NULL
          AND TRIM(CAST({did_sql} AS VARCHAR)) <> ''
        ORDER BY 1
    """
    did_df = con.execute(query).fetchdf()
    return did_df["did"].astype(str).tolist()


def normalize_input_did(raw_did: Any) -> str | None:
    if raw_did is None:
        return None
    did_value = str(raw_did).strip()
    if not did_value:
        return None
    if did_value.startswith("did:"):
        return did_value
    if "." in did_value:
        return f"did:web:{did_value}"
    return f"did:plc:{did_value}"


def build_did_lookup(raw_dids: list[str]) -> pd.DataFrame:
    did_lookup_df = pd.DataFrame({"input_did": raw_dids})
    did_lookup_df["did"] = did_lookup_df["input_did"].map(normalize_input_did)
    did_lookup_df = did_lookup_df.loc[did_lookup_df["did"].notna()].copy()
    did_lookup_df = did_lookup_df.drop_duplicates(subset=["input_did"]).reset_index(drop=True)
    return did_lookup_df


def batch_dids(dids: list[str], batch_size: int) -> list[list[str]]:
    return [dids[i : i + batch_size] for i in range(0, len(dids), batch_size)]


def chunk_file_path(batch_index: int) -> Path:
    return LONG_CHUNK_DIR / f"official_labels_long_batch_{batch_index + 1:06d}.parquet"


def discover_contiguous_completed_batches() -> set[int]:
    completed: set[int] = set()
    pattern = re.compile(r"official_labels_long_batch_(\d{6})\.parquet$")
    for path in LONG_CHUNK_DIR.glob("official_labels_long_batch_*.parquet"):
        match = pattern.search(path.name)
        if not match:
            continue
        completed.add(int(match.group(1)) - 1)
    return completed


def highest_contiguous_batch_index(completed_batches: set[int]) -> int:
    expected = 0
    while expected in completed_batches:
        expected += 1
    return expected - 1


def count_rows_in_existing_chunks() -> int:
    if not any(LONG_CHUNK_DIR.glob("official_labels_long_batch_*.parquet")):
        return 0
    con = duckdb.connect()
    try:
        query = f"""
            SELECT COUNT(*) AS n_rows
            FROM read_parquet('{path_sql(LONG_CHUNK_DIR / "official_labels_long_batch_*.parquet")}')
        """
        return int(con.execute(query).fetchone()[0])
    finally:
        con.close()


def count_rows_in_parquet_file(parquet_path: Path) -> int:
    con = duckdb.connect()
    try:
        query = f"SELECT COUNT(*) AS n_rows FROM read_parquet('{path_sql(parquet_path)}')"
        return int(con.execute(query).fetchone()[0])
    finally:
        con.close()


def retry_sleep_seconds(attempt_index: int) -> float:
    base = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2**attempt_index))
    return base + random.uniform(0.0, 0.25 * base)


def call_query_labels_page(
    *,
    uri_patterns: list[str],
    sources: list[str],
    limit: int,
    cursor: str | None = None,
) -> dict[str, Any]:
    query_params: list[tuple[str, Any]] = []
    for uri_pattern in uri_patterns:
        query_params.append(("uriPatterns", uri_pattern))
    for source in sources:
        query_params.append(("sources", source))
    query_params.append(("limit", int(limit)))
    if cursor:
        query_params.append(("cursor", cursor))

    request_url = (
        f"{PUBLIC_XRPC_BASE_URL}/xrpc/{QUERY_LABELS_NSID}?"
        f"{urlencode(query_params, doseq=True)}"
    )
    request = Request(
        url=request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "bluesky-moderation-analysis/official-labels-pipeline",
        },
        method="GET",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def call_query_labels_page_with_retry(
    *,
    uri_patterns: list[str],
    sources: list[str],
    limit: int,
    cursor: str | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt_index in range(MAX_RETRIES):
        try:
            return call_query_labels_page(
                uri_patterns=uri_patterns,
                sources=sources,
                limit=limit,
                cursor=cursor,
            )
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt_index == MAX_RETRIES - 1:
                break
            sleep_for = retry_sleep_seconds(attempt_index)
            print(
                f"  retry={attempt_index + 1}/{MAX_RETRIES - 1} "
                f"sleep={sleep_for:.1f}s error={type(exc).__name__}"
            )
            time.sleep(sleep_for)

    raise RuntimeError(
        f"Failed to call {QUERY_LABELS_NSID} after {MAX_RETRIES} attempts"
    ) from last_error


def extract_did_from_uri(uri: Any) -> str | None:
    if uri is None:
        return None
    uri_str = str(uri).strip()
    if not uri_str:
        return None
    if uri_str.startswith("did:"):
        return uri_str
    if uri_str.startswith("at://"):
        repo = uri_str[5:].split("/", 1)[0].strip()
        return repo or None
    return None


def normalize_labels_page(labels: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for label in labels:
        uri = label.get("uri")
        rows.append(
            {
                "did": extract_did_from_uri(uri),
                "uri": uri,
                "val": label.get("val"),
                "cts": label.get("cts"),
                "src": label.get("src"),
                "neg": bool(label.get("neg", False)),
            }
        )

    page_df = pd.DataFrame(rows, columns=LONG_LABEL_COLUMNS)
    if page_df.empty:
        page_df = pd.DataFrame(columns=LONG_LABEL_COLUMNS)
    page_df["cts"] = pd.to_datetime(page_df["cts"], utc=True, errors="coerce")
    page_df["neg"] = page_df["neg"].fillna(False).astype(bool)
    return page_df


def fetch_batch_labels(
    batch_did_values: list[str], batch_number: int, total_batches: int
) -> tuple[pd.DataFrame, int]:
    cursor: str | None = None
    page_number = 0
    page_frames: list[pd.DataFrame] = []

    while True:
        page_number += 1
        payload = call_query_labels_page_with_retry(
            uri_patterns=batch_did_values,
            sources=[OFFICIAL_LABELER_DID],
            limit=API_LIMIT,
            cursor=cursor,
        )
        labels = payload.get("labels", []) or []
        cursor = payload.get("cursor")

        page_df = normalize_labels_page(labels)
        if not page_df.empty:
            page_frames.append(page_df)

        print(
            f"  page {page_number:02d} [{batch_number}/{total_batches}] "
            f"labels={len(page_df):4d} cursor={'yes' if cursor else 'no'}"
        )

        if not cursor:
            break

    if not page_frames:
        return pd.DataFrame(columns=LONG_LABEL_COLUMNS), page_number

    batch_df = pd.concat(page_frames, ignore_index=True)
    batch_df = batch_df.drop_duplicates(subset=["uri", "val", "cts", "src", "neg"])
    batch_df = batch_df.sort_values(["did", "cts", "uri", "val"], na_position="last")
    batch_df = batch_df.reset_index(drop=True)
    return batch_df, page_number


def write_dataframe_to_parquet(df: pd.DataFrame, output_path: Path, view_name: str = "frame") -> None:
    con = duckdb.connect()
    try:
        con.register(view_name, df)
        con.execute(f"COPY {view_name} TO '{path_sql(output_path)}' (FORMAT PARQUET)")
    finally:
        con.close()


def write_empty_long_table(output_path: Path) -> None:
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    CAST(NULL AS VARCHAR) AS did,
                    CAST(NULL AS VARCHAR) AS uri,
                    CAST(NULL AS VARCHAR) AS val,
                    CAST(NULL AS TIMESTAMP WITH TIME ZONE) AS cts,
                    CAST(NULL AS VARCHAR) AS src,
                    CAST(NULL AS BOOLEAN) AS neg
                WHERE FALSE
            ) TO '{path_sql(output_path)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def finalize_long_table() -> None:
    chunk_glob = LONG_CHUNK_DIR / "official_labels_long_batch_*.parquet"
    if not any(LONG_CHUNK_DIR.glob("official_labels_long_batch_*.parquet")):
        write_empty_long_table(FINAL_LONG_PATH)
        return

    con = duckdb.connect()
    try:
        query = f"""
            COPY (
                SELECT DISTINCT
                    CAST(did AS VARCHAR) AS did,
                    CAST(uri AS VARCHAR) AS uri,
                    CAST(val AS VARCHAR) AS val,
                    CAST(cts AS TIMESTAMP WITH TIME ZONE) AS cts,
                    CAST(src AS VARCHAR) AS src,
                    COALESCE(CAST(neg AS BOOLEAN), FALSE) AS neg
                FROM read_parquet('{path_sql(chunk_glob)}')
            ) TO '{path_sql(FINAL_LONG_PATH)}' (FORMAT PARQUET)
        """
        con.execute(query)
    finally:
        con.close()


def load_long_labels_df() -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            SELECT did, uri, val, cts, src, neg
            FROM read_parquet('{path_sql(FINAL_LONG_PATH)}')
            """
        ).fetchdf()
    finally:
        con.close()

    if df.empty:
        empty_df = pd.DataFrame(columns=LONG_LABEL_COLUMNS)
        empty_df["cts"] = pd.to_datetime(empty_df.get("cts"), utc=True, errors="coerce")
        return empty_df

    df["cts"] = pd.to_datetime(df["cts"], utc=True, errors="coerce")
    df["neg"] = df["neg"].fillna(False).astype(bool)
    return df


def official_labels_ctes_sql() -> str:
    return f"""
        labels_base AS (
            SELECT DISTINCT
                COALESCE(
                    CAST(did AS VARCHAR),
                    CASE
                        WHEN CAST(uri AS VARCHAR) LIKE 'did:%' THEN CAST(uri AS VARCHAR)
                        WHEN CAST(uri AS VARCHAR) LIKE 'at://%' THEN split_part(substr(CAST(uri AS VARCHAR), 6), '/', 1)
                        ELSE NULL
                    END
                ) AS did,
                CAST(uri AS VARCHAR) AS uri,
                CAST(val AS VARCHAR) AS val,
                CAST(cts AS TIMESTAMP WITH TIME ZONE) AS cts,
                CAST(src AS VARCHAR) AS src,
                COALESCE(CAST(neg AS BOOLEAN), FALSE) AS neg
            FROM read_parquet('{path_sql(FINAL_LONG_PATH)}')
        ),
        official_labels AS (
            SELECT did, uri, val, cts, src, neg
            FROM labels_base
            WHERE src = '{OFFICIAL_LABELER_DID}'
              AND did IS NOT NULL
        ),
        official_non_neg_labels AS (
            SELECT did, uri, val, cts
            FROM official_labels
            WHERE NOT neg
        )
    """


def build_account_level_features(
    *, did_lookup_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    try:
        con.register("did_lookup_df", did_lookup_df[["input_did", "did"]].copy())

        account_level_query = f"""
            WITH
                did_lookup AS (
                    SELECT DISTINCT
                        CAST(input_did AS VARCHAR) AS input_did,
                        CAST(did AS VARCHAR) AS did
                    FROM did_lookup_df
                ),
                {official_labels_ctes_sql()},
                first_takedown AS (
                    SELECT did, MIN(cts) AS first_takedown_ts
                    FROM official_non_neg_labels
                    WHERE val = '!takedown'
                    GROUP BY did
                ),
                label_counts AS (
                    SELECT did, COUNT(*) AS n_official_labels
                    FROM official_non_neg_labels
                    GROUP BY did
                ),
                label_values_pre_takedown AS (
                    SELECT
                        labels.did,
                        list(DISTINCT labels.val ORDER BY labels.val) AS official_label_values
                    FROM official_non_neg_labels AS labels
                    LEFT JOIN first_takedown AS takedown USING (did)
                    WHERE takedown.first_takedown_ts IS NULL
                       OR labels.cts < takedown.first_takedown_ts
                    GROUP BY labels.did
                )
            SELECT
                lookup.did,
                lookup.input_did,
                CASE WHEN takedown.first_takedown_ts IS NOT NULL THEN 1 ELSE 0 END AS has_takedown,
                takedown.first_takedown_ts,
                COALESCE(counts.n_official_labels, 0) AS n_official_labels,
                COALESCE(values.official_label_values, []::VARCHAR[]) AS official_label_values
            FROM did_lookup AS lookup
            LEFT JOIN first_takedown AS takedown USING (did)
            LEFT JOIN label_counts AS counts USING (did)
            LEFT JOIN label_values_pre_takedown AS values USING (did)
            ORDER BY lookup.input_did
        """
        account_df = con.execute(account_level_query).fetchdf()
        account_df["first_takedown_ts"] = pd.to_datetime(
            account_df["first_takedown_ts"], utc=True, errors="coerce"
        )
        account_df["has_takedown"] = account_df["has_takedown"].fillna(0).astype(int)
        account_df["n_official_labels"] = account_df["n_official_labels"].fillna(0).astype(int)
        account_df["official_label_values"] = account_df["official_label_values"].apply(normalize_list_cell)

        other_labels_counts_df = con.execute(
            f"""
                WITH
                    {official_labels_ctes_sql()}
                SELECT
                    val,
                    COUNT(*) AS label_count
                FROM official_non_neg_labels
                WHERE val <> '!takedown'
                GROUP BY val
                ORDER BY val
            """
        ).fetchdf()
    finally:
        con.close()

    return account_df, other_labels_counts_df


def write_account_level_and_enriched_outputs(
    *,
    account_level_df: pd.DataFrame,
    did_column_name: str,
) -> None:
    con = duckdb.connect()
    try:
        con.register("account_level_df", account_level_df)
        con.execute(
            f"""
            COPY (
                SELECT
                    CAST(did AS VARCHAR) AS did,
                    CAST(input_did AS VARCHAR) AS input_did,
                    CAST(has_takedown AS INTEGER) AS has_takedown,
                    CAST(first_takedown_ts AS TIMESTAMP WITH TIME ZONE) AS first_takedown_ts,
                    CAST(n_official_labels AS INTEGER) AS n_official_labels,
                    CAST(official_label_values AS VARCHAR[]) AS official_label_values
                FROM account_level_df
            ) TO '{path_sql(ACCOUNT_LEVEL_PATH)}' (FORMAT PARQUET)
            """
        )

        did_sql = quote_identifier(did_column_name)
        did_alias_sql = ", a.did AS did"
        input_did_alias_sql = ""
        if did_column_name != "did":
            input_did_alias_sql = f", CAST(f.{did_sql} AS VARCHAR) AS input_did"
        else:
            did_alias_sql = ", a.did AS normalized_did"

        enriched_query = f"""
            COPY (
                SELECT
                    f.*
                    {input_did_alias_sql}
                    {did_alias_sql},
                    CAST(a.has_takedown AS INTEGER) AS has_takedown,
                    CAST(a.first_takedown_ts AS TIMESTAMP WITH TIME ZONE) AS first_takedown_ts,
                    CAST(a.n_official_labels AS INTEGER) AS n_official_labels,
                    CAST(a.official_label_values AS VARCHAR[]) AS official_label_values
                FROM read_parquet('{path_sql(INPUT_PARQUET_PATH)}') AS f
                LEFT JOIN account_level_df AS a
                    ON CAST(f.{did_sql} AS VARCHAR) = a.input_did
            ) TO '{path_sql(ENRICHED_PATH)}' (FORMAT PARQUET)
        """
        con.execute(enriched_query)
    finally:
        con.close()


def print_final_summary(
    *,
    raw_dids: list[str],
    account_level_df: pd.DataFrame,
    other_labels_counts_df: pd.DataFrame,
) -> None:
    n_unique_dids = len(raw_dids)
    n_dids_with_labels = int((account_level_df["n_official_labels"] > 0).sum())
    n_dids_with_takedown = int(account_level_df["has_takedown"].sum())

    print()
    print("Final summary")
    print(f"- number of unique DIDs in input: {n_unique_dids}")
    print(f"- number of DIDs with at least one official label: {n_dids_with_labels}")
    print(f"- number of DIDs with !takedown: {n_dids_with_takedown}")
    print("- official labels other than !takedown:")
    if other_labels_counts_df.empty:
        print("  none")
    else:
        for row in other_labels_counts_df.itertuples(index=False):
            print(f"  {row.val}: {int(row.label_count)}")


def run_pipeline() -> None:
    ensure_output_dirs()

    if not INPUT_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PARQUET_PATH}")

    con = duckdb.connect()
    try:
        did_column_name = discover_did_column_name(con, INPUT_PARQUET_PATH)
        raw_dids = extract_unique_dids(con, INPUT_PARQUET_PATH, did_column_name)
    finally:
        con.close()

    did_lookup_df = build_did_lookup(raw_dids)
    query_dids = did_lookup_df["did"].drop_duplicates().astype(str).tolist()
    batches = batch_dids(query_dids, BATCH_SIZE)
    total_batches = len(batches)

    checkpoint = parse_checkpoint()
    checkpoint_batch_size = checkpoint.get("batch_size")
    if checkpoint_batch_size is not None and int(checkpoint_batch_size) != BATCH_SIZE:
        raise ValueError(
            "Checkpoint batch size does not match the current BATCH_SIZE. "
            "Delete the existing checkpoint/chunk files or restore the previous batch size."
        )

    completed_batches = discover_contiguous_completed_batches()
    last_completed_batch_index = highest_contiguous_batch_index(completed_batches)

    checkpoint_last = int(checkpoint.get("last_completed_batch_index", -1))
    if checkpoint_last > last_completed_batch_index:
        last_completed_batch_index = checkpoint_last

    total_labels_collected_so_far = count_rows_in_existing_chunks()

    write_checkpoint(
        did_column_name=did_column_name,
        total_batches=total_batches,
        last_completed_batch_index=last_completed_batch_index,
        total_labels_collected_so_far=total_labels_collected_so_far,
        total_unique_dids=len(raw_dids),
        extra={"status": "running", "total_query_dids": len(query_dids)},
    )

    start_batch_index = last_completed_batch_index + 1
    if start_batch_index < total_batches:
        print(
            f"Resuming label collection from batch {start_batch_index + 1} of {total_batches}"
        )
    else:
        print(f"All {total_batches} batches are already collected; rebuilding final outputs.")

    for batch_index in range(start_batch_index, total_batches):
        batch_number = batch_index + 1
        batch_path = chunk_file_path(batch_index)

        if batch_path.exists():
            batch_labels_written = count_rows_in_parquet_file(batch_path)
            write_checkpoint(
                did_column_name=did_column_name,
                total_batches=total_batches,
                last_completed_batch_index=batch_index,
                total_labels_collected_so_far=total_labels_collected_so_far,
                total_unique_dids=len(raw_dids),
                extra={"status": "running"},
            )
            print(
                f"[{batch_number:04d}/{total_batches:04d}] "
                f"existing chunk detected, skipping API call "
                f"(labels={batch_labels_written:4d})"
            )
            continue

        batch_did_values = batches[batch_index]
        batch_df, page_count = fetch_batch_labels(
            batch_did_values=batch_did_values,
            batch_number=batch_number,
            total_batches=total_batches,
        )

        if batch_df.empty:
            # No file is written for empty batches. The checkpoint is still advanced.
            batch_labels_written = 0
        else:
            write_dataframe_to_parquet(batch_df, batch_path, view_name="batch_df")
            batch_labels_written = len(batch_df)

        total_labels_collected_so_far += batch_labels_written
        write_checkpoint(
            did_column_name=did_column_name,
            total_batches=total_batches,
            last_completed_batch_index=batch_index,
            total_labels_collected_so_far=total_labels_collected_so_far,
            total_unique_dids=len(raw_dids),
            extra={"status": "running"},
        )

        print(
            f"[{batch_number:04d}/{total_batches:04d}] "
            f"dids={len(batch_did_values):3d} "
            f"labels={batch_labels_written:4d} "
            f"cumulative={total_labels_collected_so_far:7d} "
            f"pages={page_count:02d}"
        )

    finalize_long_table()
    account_level_df, other_labels_counts_df = build_account_level_features(
        did_lookup_df=did_lookup_df
    )
    write_account_level_and_enriched_outputs(
        account_level_df=account_level_df,
        did_column_name=did_column_name,
    )

    write_checkpoint(
        did_column_name=did_column_name,
        total_batches=total_batches,
        last_completed_batch_index=total_batches - 1,
        total_labels_collected_so_far=total_labels_collected_so_far,
        total_unique_dids=len(raw_dids),
        extra={
            "status": "completed",
            "completed_at": utc_now_iso(),
            "final_long_path": str(FINAL_LONG_PATH),
            "account_level_path": str(ACCOUNT_LEVEL_PATH),
            "enriched_path": str(ENRICHED_PATH),
        },
    )

    print_final_summary(
        raw_dids=raw_dids,
        account_level_df=account_level_df,
        other_labels_counts_df=other_labels_counts_df,
    )


# %%
if __name__ == "__main__":
    run_pipeline()

### OUTPUT:
## Final summary
#- number of unique DIDs in input: 193556
#- number of DIDs with at least one official label: 3045
#- number of DIDs with !takedown: 2649
#- official labels other than !takedown:
#  !hide: 9
#  !unspecced-takedown: 5
#  impersonation: 42
#  intolerant: 41
#  needs-review: 58
#  nudity: 6
#  porn: 5
#  rude: 103
#  scam: 1
#  sexual: 63
#  sexual-figurative: 8
#  spam: 153
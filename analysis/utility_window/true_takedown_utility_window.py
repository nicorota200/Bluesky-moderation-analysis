""" data from datasets/model/labeller """   
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import duckdb
import math
import numpy as np
import pandas as pd


"""
Positive-only utility-window analysis anchored on the true takedown event.

This module uses the column names verified in the current parquet schemas:
- labels parquet: `did`, `has_takedown`, `first_takedown_ts`
- fingerprints parquet: `did`, `input_did`, `has_takedown`, `first_takedown_ts`
- posts parquet: `did_norm`, `post_time`, `post_uri`

"""


SECONDS_PER_DAY = 86_400.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATION_START = pd.Timestamp("2024-06-01", tz="UTC")
DEFAULT_LATE_GRID = (1, 2, 3, 5, 7, 10, 12, 14, 16, 18, 21, 30)
DEFAULT_EARLY_GRID = (1, 2, 3, 5, 7)
DEFAULT_UNSUPPORTED_SHARE_THRESHOLD = 0.10
DEFAULT_UNSUPPORTED_COUNT_THRESHOLD = 100
DEFAULT_MIN_UNSUPPORTED_WITH_POSTS_FOR_EARLY_ESTIMATION = 30

DEFAULT_FINGERPRINTS_PARQUET = (
    REPO_ROOT
    / "datasets"
    / "model"
    / "labeller"
    / "account_level_labels"
    / "account_level_labels_in_time_window"
    / "fingerprints_enriched_with_official_labels.parquet"
)
DEFAULT_LABELS_PARQUET = (
    REPO_ROOT
    / "datasets"
    / "model"
    / "labeller"
    / "account_level_labels"
    / "account_level_labels_in_time_window"
    / "official_labels_account_level.parquet"
)
DEFAULT_WINDOW_POSTS_PARQUET = (
    REPO_ROOT
    / "datasets"
    / "model"
    / "labeller"
    / "post_level_labels"
    / "window_posts.parquet"
)

LABELS_COLUMN_MAP = {
    "did": "did",
    "has_takedown": "has_takedown",
    "first_takedown_ts": "first_takedown_ts",
}
FINGERPRINTS_COLUMN_MAP = {
    "did": "did",
    "input_did": "input_did",
    "has_takedown": "has_takedown",
    "first_takedown_ts": "first_takedown_ts",
}
POSTS_COLUMN_MAP = {
    "did": "did_norm",
    "post_time": "post_time",
    "post_uri": "post_uri",
}


@dataclass
class PositiveAnchorBuildResult:
    """Outputs from the true takedown anchor construction step."""

    positive_anchors: pd.DataFrame
    positive_consistency_diagnostics: pd.DataFrame
    fingerprint_mismatch_details: pd.DataFrame


@dataclass
class PreEventPostsExtractionResult:
    """Outputs from the pre-event post extraction step."""

    pre_event_posts: pd.DataFrame
    account_diagnostics: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class LateWindowRecommendation:
    """Primary late-window recommendation."""

    late_window: int | None
    continuous_q75_of_q80: float | None
    n_positive_total: int
    n_long_history_total: int
    n_long_history_with_posts: int
    n_accounts_used_for_estimation: int
    long_history_threshold_days: int
    rule: str
    reason: str


@dataclass(frozen=True)
class EarlyWindowRecommendation:
    """Decision about whether an early window is needed."""

    needs_early_window: bool
    triggered_for_evaluation: bool
    accepted: bool
    early_window: int | None
    candidate_early_window: int | None
    continuous_q75_of_q80_short: float | None
    n_positive_total: int
    n_unsupported_late: int
    unsupported_share: float | None
    n_unsupported_with_posts: int
    n_accounts_used_for_estimation: int
    minimum_accounts_required: int
    late_window: int | None
    reason: str


@dataclass
class TrueTakedownUtilityWindowAnalysisResult:
    """End-to-end result for the true takedown utility-window analysis."""

    positive_anchors: pd.DataFrame
    pre_event_posts: pd.DataFrame
    pre_event_post_account_diagnostics: pd.DataFrame
    account_distance_quantiles: pd.DataFrame
    positive_consistency_diagnostics: pd.DataFrame
    fingerprint_mismatch_details: pd.DataFrame
    late_grid_diagnostics: pd.DataFrame
    late_recommendation: LateWindowRecommendation
    early_recommendation: EarlyWindowRecommendation
    summary: dict[str, Any]


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    """Convert a scalar timestamp-like value to UTC."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def to_utc_series(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to UTC, treating naive values as UTC."""

    return pd.to_datetime(series, errors="coerce", utc=True)


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier safely."""

    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'


def sql_string_literal(value: str | Path) -> str:
    """Quote a SQL string literal safely."""

    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def connect_duckdb_utc() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection pinned to UTC."""

    connection = duckdb.connect()
    connection.execute("SET TimeZone = 'UTC'")
    return connection


def list_parquet_columns(parquet_path: str | Path) -> list[str]:
    """Read parquet column names."""

    connection = connect_duckdb_utc()
    try:
        query = (
            "DESCRIBE SELECT * "
            f"FROM read_parquet({sql_string_literal(Path(parquet_path))})"
        )
        schema = connection.execute(query).fetchdf()
    finally:
        connection.close()
    return schema.iloc[:, 0].astype(str).tolist()


def validate_expected_columns(
    parquet_path: str | Path,
    expected_columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Fail clearly if the parquet schema changed."""

    available = set(list_parquet_columns(parquet_path))
    missing = [column for column in expected_columns if column not in available]
    if missing:
        raise ValueError(
            f"{dataset_name} parquet is missing expected columns {missing}. "
            f"Available columns: {sorted(available)}"
        )


def read_parquet_columns_to_frame(
    parquet_path: str | Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Load only the requested parquet columns into pandas."""

    unique_columns = list(dict.fromkeys(columns))
    if not unique_columns:
        raise ValueError("At least one column must be selected from the parquet file.")

    connection = connect_duckdb_utc()
    try:
        select_sql = ", ".join(quote_identifier(column) for column in unique_columns)
        query = (
            f"SELECT {select_sql} "
            f"FROM read_parquet({sql_string_literal(Path(parquet_path))})"
        )
        return connection.execute(query).fetchdf()
    finally:
        connection.close()


def _normalize_grid(grid: Sequence[int], *, name: str) -> tuple[int, ...]:
    normalized = sorted({int(window) for window in grid if int(window) > 0})
    if not normalized:
        raise ValueError(f"{name} must contain at least one positive integer window.")
    return tuple(normalized)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _coerce_truthy(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(False, index=series.index, dtype=bool)
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask] != 0

    text_mask = ~numeric_mask & series.notna()
    if text_mask.any():
        text = series.loc[text_mask].astype(str).str.strip().str.lower()
        result.loc[text_mask] = text.isin({"true", "t", "yes", "y", "1"})
    return result


def _choose_smallest_grid_window_at_or_above(
    value: float | None,
    grid: Sequence[int],
) -> int | None:
    if value is None or math.isnan(value):
        return None
    normalized_grid = _normalize_grid(grid, name="grid")
    if value <= normalized_grid[0]:
        return normalized_grid[0]
    for candidate in normalized_grid:
        if candidate >= value:
            return candidate
    return normalized_grid[-1]


def _build_diagnostics_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["diagnostic", "value", "details"])


def _merge_post_account_diagnostics(
    positive_anchors: pd.DataFrame,
    account_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    merged = positive_anchors.merge(
        account_diagnostics,
        on="did",
        how="left",
        suffixes=("_base", ""),
    )
    for column in (
        "n_pre_event_posts",
        "first_pre_event_post_time",
        "last_pre_event_post_time",
    ):
        base_column = f"{column}_base"
        if base_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].combine_first(merged[base_column])
            else:
                merged[column] = merged[base_column]
            merged = merged.drop(columns=[base_column])

    merged["n_pre_event_posts"] = (
        pd.to_numeric(merged["n_pre_event_posts"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    merged["has_pre_event_posts"] = merged["n_pre_event_posts"] > 0
    return merged


def build_true_takedown_positive_anchors(
    labels_parquet_path: str | Path = DEFAULT_LABELS_PARQUET,
    fingerprints_parquet_path: str | Path = DEFAULT_FINGERPRINTS_PARQUET,
    observation_start: Any = DEFAULT_OBSERVATION_START,
) -> PositiveAnchorBuildResult:
    """
    Build the positive anchor table using the first recoverable true takedown time.

    The current labels parquet is already account-level and exposes `has_takedown`
    plus `first_takedown_ts`, so those verified columns are used directly.
    """

    observation_start_utc = to_utc_timestamp(observation_start)

    validate_expected_columns(
        labels_parquet_path,
        LABELS_COLUMN_MAP.values(),
        dataset_name="labels",
    )
    validate_expected_columns(
        fingerprints_parquet_path,
        FINGERPRINTS_COLUMN_MAP.values(),
        dataset_name="fingerprints",
    )

    labels = read_parquet_columns_to_frame(
        labels_parquet_path,
        list(LABELS_COLUMN_MAP.values()),
    )
    labels["did"] = labels[LABELS_COLUMN_MAP["did"]].astype("string").str.strip()
    labels["takedown_time"] = to_utc_series(
        labels[LABELS_COLUMN_MAP["first_takedown_ts"]]
    )
    labels["has_takedown"] = _coerce_truthy(labels[LABELS_COLUMN_MAP["has_takedown"]])

    positive_rows = labels.loc[
        labels["did"].notna()
        & labels["did"].ne("")
        & labels["has_takedown"]
        & labels["takedown_time"].notna()
    ].copy()
    if positive_rows.empty:
        raise ValueError(
            "Unable to recover any positive account with a valid takedown timestamp "
            f"from {labels_parquet_path}."
        )

    positive_anchors = (
        positive_rows.sort_values(["did", "takedown_time"])
        .drop_duplicates(subset=["did"], keep="first")
        .loc[:, ["did", "takedown_time"]]
        .reset_index(drop=True)
    )
    positive_anchors["obs_days"] = (
        (positive_anchors["takedown_time"] - observation_start_utc)
        .dt.total_seconds()
        / SECONDS_PER_DAY
    )
    positive_anchors["has_pre_event_posts"] = False
    positive_anchors["n_pre_event_posts"] = 0
    positive_anchors["first_pre_event_post_time"] = pd.NaT
    positive_anchors["last_pre_event_post_time"] = pd.NaT

    fingerprints = read_parquet_columns_to_frame(
        fingerprints_parquet_path,
        list(FINGERPRINTS_COLUMN_MAP.values()),
    )
    fingerprints["did"] = (
        fingerprints[FINGERPRINTS_COLUMN_MAP["did"]].astype("string").str.strip()
    )
    fingerprints["input_did"] = (
        fingerprints[FINGERPRINTS_COLUMN_MAP["input_did"]].astype("string").str.strip()
    )
    fingerprints["has_takedown"] = _coerce_truthy(
        fingerprints[FINGERPRINTS_COLUMN_MAP["has_takedown"]]
    )
    fingerprints["takedown_time"] = to_utc_series(
        fingerprints[FINGERPRINTS_COLUMN_MAP["first_takedown_ts"]]
    )

    fingerprint_positives = (
        fingerprints.loc[
            fingerprints["did"].notna()
            & fingerprints["did"].ne("")
            & (
                fingerprints["has_takedown"]
                | fingerprints["takedown_time"].notna()
            )
        ]
        .copy()
        .drop_duplicates(subset=["did"])
    )

    label_positive_set = set(positive_anchors["did"].tolist())
    fingerprint_positive_set = set(fingerprint_positives["did"].tolist())
    fingerprint_only = sorted(fingerprint_positive_set - label_positive_set)
    labels_only = sorted(label_positive_set - fingerprint_positive_set)

    mismatch_frames: list[pd.DataFrame] = []
    if fingerprint_only:
        mismatch_frames.append(
            fingerprint_positives.loc[
                fingerprint_positives["did"].isin(fingerprint_only),
                ["did", "input_did", "takedown_time"],
            ]
            .assign(
                mismatch_type="fingerprint_positive_missing_recoverable_takedown_time"
            )
        )
    if labels_only:
        mismatch_frames.append(
            positive_anchors.loc[
                positive_anchors["did"].isin(labels_only),
                ["did", "takedown_time"],
            ]
            .assign(
                input_did=pd.NA,
                mismatch_type="labels_positive_missing_from_fingerprints",
            )
        )

    if mismatch_frames:
        fingerprint_mismatch_details = pd.concat(mismatch_frames, ignore_index=True)
    else:
        fingerprint_mismatch_details = pd.DataFrame(
            columns=["did", "input_did", "takedown_time", "mismatch_type"]
        )

    positive_consistency_diagnostics = _build_diagnostics_frame(
        [
            {
                "diagnostic": "labels_positive_accounts_with_recoverable_takedown_time",
                "value": int(len(positive_anchors)),
                "details": "Main analysis universe: positives with recoverable T_i from the labels parquet.",
            },
            {
                "diagnostic": "positive_accounts_before_observation_start",
                "value": int((positive_anchors["obs_days"] < 0).sum()),
                "details": "These accounts have T_i earlier than A and therefore no observable pre-event history.",
            },
            {
                "diagnostic": "fingerprints_positive_accounts",
                "value": int(len(fingerprint_positive_set)),
                "details": "Positive accounts flagged by the fingerprints parquet.",
            },
            {
                "diagnostic": "labels_vs_fingerprints_overlap",
                "value": int(len(label_positive_set & fingerprint_positive_set)),
                "details": "Accounts positive in both datasets.",
            },
            {
                "diagnostic": "fingerprints_missing_recoverable_takedown_time",
                "value": int(len(fingerprint_only)),
                "details": "Positive in fingerprints but missing from the labels-driven anchor table.",
            },
            {
                "diagnostic": "labels_missing_from_fingerprints",
                "value": int(len(labels_only)),
                "details": "Positive in the labels-driven anchor table but not positive in fingerprints.",
            },
        ]
    )

    return PositiveAnchorBuildResult(
        positive_anchors=positive_anchors,
        positive_consistency_diagnostics=positive_consistency_diagnostics,
        fingerprint_mismatch_details=fingerprint_mismatch_details,
    )


def extract_pre_event_posts_for_positives(
    posts_parquet_path: str | Path,
    positive_anchors: pd.DataFrame,
    observation_start: Any = DEFAULT_OBSERVATION_START,
) -> PreEventPostsExtractionResult:
    """Extract only positive-account posts satisfying A <= post_time < T_i."""

    observation_start_utc = to_utc_timestamp(observation_start)
    validate_expected_columns(
        posts_parquet_path,
        POSTS_COLUMN_MAP.values(),
        dataset_name="posts",
    )

    if positive_anchors.empty:
        empty_posts = pd.DataFrame(
            columns=["did", "post_time", "post_uri", "takedown_time", "distance_days"]
        )
        empty_account = pd.DataFrame(
            columns=[
                "did",
                "n_pre_event_posts",
                "first_pre_event_post_time",
                "last_pre_event_post_time",
            ]
        )
        return PreEventPostsExtractionResult(
            pre_event_posts=empty_posts,
            account_diagnostics=empty_account,
            summary={
                "n_positive_accounts": 0,
                "n_positive_accounts_with_posts": 0,
                "n_pre_event_posts": 0,
            },
        )

    anchors = positive_anchors.loc[:, ["did", "takedown_time"]].copy()
    anchors["did"] = anchors["did"].astype("string")
    anchors["takedown_time"] = to_utc_series(anchors["takedown_time"])

    connection = connect_duckdb_utc()
    try:
        connection.register("positive_anchors", anchors)
        query = f"""
            SELECT
                a.did AS did,
                CAST(p.{quote_identifier(POSTS_COLUMN_MAP["post_time"])} AS TIMESTAMPTZ) AS post_time,
                p.{quote_identifier(POSTS_COLUMN_MAP["post_uri"])} AS post_uri,
                a.takedown_time AS takedown_time
            FROM read_parquet({sql_string_literal(Path(posts_parquet_path))}) AS p
            INNER JOIN positive_anchors AS a
                ON CAST(p.{quote_identifier(POSTS_COLUMN_MAP["did"])} AS VARCHAR) = a.did
            WHERE CAST(p.{quote_identifier(POSTS_COLUMN_MAP["post_time"])} AS TIMESTAMPTZ)
                    >= TIMESTAMPTZ {sql_string_literal(observation_start_utc.isoformat())}
              AND CAST(p.{quote_identifier(POSTS_COLUMN_MAP["post_time"])} AS TIMESTAMPTZ)
                    < a.takedown_time
        """
        pre_event_posts = connection.execute(query).fetchdf()
    finally:
        connection.close()

    if pre_event_posts.empty:
        empty_account = pd.DataFrame(
            columns=[
                "did",
                "n_pre_event_posts",
                "first_pre_event_post_time",
                "last_pre_event_post_time",
            ]
        )
        return PreEventPostsExtractionResult(
            pre_event_posts=pd.DataFrame(
                columns=["did", "post_time", "post_uri", "takedown_time", "distance_days"]
            ),
            account_diagnostics=empty_account,
            summary={
                "n_positive_accounts": int(len(positive_anchors)),
                "n_positive_accounts_with_posts": 0,
                "n_pre_event_posts": 0,
            },
        )

    pre_event_posts["did"] = pre_event_posts["did"].astype("string")
    pre_event_posts["post_time"] = to_utc_series(pre_event_posts["post_time"])
    pre_event_posts["takedown_time"] = to_utc_series(pre_event_posts["takedown_time"])
    pre_event_posts["distance_days"] = (
        (pre_event_posts["takedown_time"] - pre_event_posts["post_time"])
        .dt.total_seconds()
        / SECONDS_PER_DAY
    )
    pre_event_posts = pre_event_posts.loc[
        pre_event_posts["distance_days"] > 0
    ].copy()
    pre_event_posts = pre_event_posts.sort_values(["did", "post_time"]).reset_index(drop=True)

    account_diagnostics = (
        pre_event_posts.groupby("did", as_index=False)
        .agg(
            n_pre_event_posts=("did", "size"),
            first_pre_event_post_time=("post_time", "min"),
            last_pre_event_post_time=("post_time", "max"),
        )
        .sort_values("did")
        .reset_index(drop=True)
    )

    return PreEventPostsExtractionResult(
        pre_event_posts=pre_event_posts,
        account_diagnostics=account_diagnostics,
        summary={
            "n_positive_accounts": int(len(positive_anchors)),
            "n_positive_accounts_with_posts": int(account_diagnostics["did"].nunique()),
            "n_pre_event_posts": int(len(pre_event_posts)),
        },
    )


def attach_pre_event_post_diagnostics(
    positive_anchors: pd.DataFrame,
    account_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per-account post diagnostics to the positive anchor table."""

    return _merge_post_account_diagnostics(positive_anchors, account_diagnostics)


def compute_account_distance_quantiles(
    pre_event_posts: pd.DataFrame,
    quantiles: Sequence[float] = (0.80,),
) -> pd.DataFrame:
    """Compute account-level quantiles of pre-event post distances."""

    normalized_quantiles = tuple(sorted({float(quantile) for quantile in quantiles}))
    if not normalized_quantiles:
        raise ValueError("At least one quantile must be requested.")
    if pre_event_posts.empty:
        columns = ["did", "n_pre_event_posts"] + [
            f"q{int(round(quantile * 100))}" for quantile in normalized_quantiles
        ]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for did, group in pre_event_posts.groupby("did", sort=True):
        distances = group["distance_days"].astype(float)
        row: dict[str, Any] = {
            "did": did,
            "n_pre_event_posts": int(len(group)),
        }
        for quantile in normalized_quantiles:
            row[f"q{int(round(quantile * 100))}"] = float(distances.quantile(quantile))
        rows.append(row)
    return pd.DataFrame(rows)


def build_late_grid_diagnostics(
    positive_anchors: pd.DataFrame,
    pre_event_posts: pd.DataFrame,
    late_grid: Sequence[int] = DEFAULT_LATE_GRID,
) -> pd.DataFrame:
    """Build diagnostics for every late-grid window."""

    normalized_grid = _normalize_grid(late_grid, name="late_grid")
    if "n_pre_event_posts" not in positive_anchors.columns:
        raise ValueError(
            "positive_anchors must contain n_pre_event_posts before late-grid diagnostics are computed."
        )

    total_positives = int(len(positive_anchors))
    total_pre_event_posts = int(len(pre_event_posts))
    previous_global_share = 0.0
    rows: list[dict[str, Any]] = []

    for window in normalized_grid:
        supported_mask = positive_anchors["obs_days"] >= float(window)
        supported_with_posts = supported_mask & positive_anchors["has_pre_event_posts"]

        if supported_with_posts.any():
            supported_accounts = positive_anchors.loc[
                supported_with_posts, ["did", "n_pre_event_posts"]
            ].copy()
            captured_counts = (
                pre_event_posts.loc[pre_event_posts["distance_days"] <= float(window)]
                .groupby("did")
                .size()
                .rename("n_captured")
            )
            supported_accounts = supported_accounts.merge(
                captured_counts,
                left_on="did",
                right_index=True,
                how="left",
            )
            supported_accounts["n_captured"] = (
                supported_accounts["n_captured"].fillna(0).astype(int)
            )
            supported_accounts["capture_share"] = (
                supported_accounts["n_captured"]
                / supported_accounts["n_pre_event_posts"]
            )
            median_capture = float(supported_accounts["capture_share"].median())
            p75_capture = float(supported_accounts["capture_share"].quantile(0.75))
        else:
            median_capture = np.nan
            p75_capture = np.nan

        if total_pre_event_posts > 0:
            global_post_capture_share = float(
                (pre_event_posts["distance_days"] <= float(window)).mean()
            )
            incremental_share = global_post_capture_share - previous_global_share
            previous_global_share = global_post_capture_share
        else:
            global_post_capture_share = np.nan
            incremental_share = np.nan

        rows.append(
            {
                "window_days": int(window),
                "n_positive_total": total_positives,
                "n_positive_supported": int(supported_mask.sum()),
                "support_share": (
                    float(supported_mask.mean()) if total_positives else np.nan
                ),
                "n_positive_with_posts_supported": int(supported_with_posts.sum()),
                "median_account_capture_share": median_capture,
                "p75_account_capture_share": p75_capture,
                "global_post_capture_share": global_post_capture_share,
                "incremental_global_post_capture_share": incremental_share,
            }
        )

    return pd.DataFrame(rows)


def select_late_window(
    positive_anchors: pd.DataFrame,
    account_distance_quantiles: pd.DataFrame,
    late_grid: Sequence[int] = DEFAULT_LATE_GRID,
) -> LateWindowRecommendation:
    """Select the late window using Q75 of per-account q80 on long-history positives."""

    normalized_grid = _normalize_grid(late_grid, name="late_grid")
    long_history_threshold_days = max(normalized_grid)
    total_positives = int(len(positive_anchors))

    if "q80" not in account_distance_quantiles.columns:
        raise ValueError("account_distance_quantiles must contain a 'q80' column.")

    long_history = positive_anchors.loc[
        positive_anchors["obs_days"] >= float(long_history_threshold_days)
    ].copy()
    long_history_with_posts = long_history.loc[
        long_history["has_pre_event_posts"]
    ].copy()

    if account_distance_quantiles.empty or long_history_with_posts.empty:
        return LateWindowRecommendation(
            late_window=None,
            continuous_q75_of_q80=None,
            n_positive_total=total_positives,
            n_long_history_total=int(len(long_history)),
            n_long_history_with_posts=int(len(long_history_with_posts)),
            n_accounts_used_for_estimation=0,
            long_history_threshold_days=long_history_threshold_days,
            rule="smallest late_grid window >= Q75(q80_i)",
            reason=(
                "Late window unavailable because there are no long-history positives "
                "with at least one pre-event post."
            ),
        )

    estimator_frame = long_history_with_posts.merge(
        account_distance_quantiles.loc[:, ["did", "q80"]],
        on="did",
        how="inner",
    )
    valid_q80 = estimator_frame["q80"].dropna()
    if valid_q80.empty:
        return LateWindowRecommendation(
            late_window=None,
            continuous_q75_of_q80=None,
            n_positive_total=total_positives,
            n_long_history_total=int(len(long_history)),
            n_long_history_with_posts=int(len(long_history_with_posts)),
            n_accounts_used_for_estimation=0,
            long_history_threshold_days=long_history_threshold_days,
            rule="smallest late_grid window >= Q75(q80_i)",
            reason="Late window unavailable because q80_i is undefined for all long-history positives.",
        )

    continuous_q75 = float(valid_q80.quantile(0.75))
    late_window = _choose_smallest_grid_window_at_or_above(
        continuous_q75,
        normalized_grid,
    )
    if late_window is None:
        raise RuntimeError("Late-window selection failed despite valid q80 estimates.")

    return LateWindowRecommendation(
        late_window=int(late_window),
        continuous_q75_of_q80=continuous_q75,
        n_positive_total=total_positives,
        n_long_history_total=int(len(long_history)),
        n_long_history_with_posts=int(len(long_history_with_posts)),
        n_accounts_used_for_estimation=int(len(valid_q80)),
        long_history_threshold_days=long_history_threshold_days,
        rule="smallest late_grid window >= Q75(q80_i)",
        reason=(
            "Late window selected from long-history positives only, using the 75th percentile "
            "of the per-account q80 distances."
        ),
    )


def evaluate_early_window_need_and_selection(
    positive_anchors: pd.DataFrame,
    account_distance_quantiles: pd.DataFrame,
    late_window: int | None,
    early_grid: Sequence[int] = DEFAULT_EARLY_GRID,
    unsupported_share_threshold: float = DEFAULT_UNSUPPORTED_SHARE_THRESHOLD,
    unsupported_count_threshold: int = DEFAULT_UNSUPPORTED_COUNT_THRESHOLD,
    minimum_unsupported_with_posts_for_early_estimation: int = (
        DEFAULT_MIN_UNSUPPORTED_WITH_POSTS_FOR_EARLY_ESTIMATION
    ),
) -> EarlyWindowRecommendation:
    """Decide whether an early window is needed and select it if accepted."""

    normalized_grid = _normalize_grid(early_grid, name="early_grid")
    total_positives = int(len(positive_anchors))

    if "q80" not in account_distance_quantiles.columns:
        raise ValueError("account_distance_quantiles must contain a 'q80' column.")

    if late_window is None:
        return EarlyWindowRecommendation(
            needs_early_window=False,
            triggered_for_evaluation=False,
            accepted=False,
            early_window=None,
            candidate_early_window=None,
            continuous_q75_of_q80_short=None,
            n_positive_total=total_positives,
            n_unsupported_late=0,
            unsupported_share=None,
            n_unsupported_with_posts=0,
            n_accounts_used_for_estimation=0,
            minimum_accounts_required=minimum_unsupported_with_posts_for_early_estimation,
            late_window=None,
            reason="Early window not evaluated because the late window could not be estimated.",
        )

    unsupported_late = positive_anchors.loc[
        positive_anchors["obs_days"] < float(late_window)
    ].copy()
    n_unsupported_late = int(len(unsupported_late))
    unsupported_share = (
        n_unsupported_late / total_positives if total_positives else np.nan
    )
    unsupported_with_posts = unsupported_late.loc[
        unsupported_late["has_pre_event_posts"]
    ].copy()
    n_unsupported_with_posts = int(len(unsupported_with_posts))

    triggered = (
        (unsupported_share >= unsupported_share_threshold)
        or (n_unsupported_late >= int(unsupported_count_threshold))
    )
    if not triggered:
        return EarlyWindowRecommendation(
            needs_early_window=False,
            triggered_for_evaluation=False,
            accepted=False,
            early_window=None,
            candidate_early_window=None,
            continuous_q75_of_q80_short=None,
            n_positive_total=total_positives,
            n_unsupported_late=n_unsupported_late,
            unsupported_share=_maybe_float(unsupported_share),
            n_unsupported_with_posts=n_unsupported_with_posts,
            n_accounts_used_for_estimation=0,
            minimum_accounts_required=minimum_unsupported_with_posts_for_early_estimation,
            late_window=int(late_window),
            reason=(
                "Early window not needed because the unsupported-late trigger was not met: "
                f"unsupported_share={unsupported_share:.4f}, "
                f"n_unsupported_late={n_unsupported_late}."
            ),
        )

    if n_unsupported_with_posts == 0:
        return EarlyWindowRecommendation(
            needs_early_window=False,
            triggered_for_evaluation=True,
            accepted=False,
            early_window=None,
            candidate_early_window=None,
            continuous_q75_of_q80_short=None,
            n_positive_total=total_positives,
            n_unsupported_late=n_unsupported_late,
            unsupported_share=_maybe_float(unsupported_share),
            n_unsupported_with_posts=0,
            n_accounts_used_for_estimation=0,
            minimum_accounts_required=minimum_unsupported_with_posts_for_early_estimation,
            late_window=int(late_window),
            reason=(
                "Early window trigger was met, but no unsupported-late positives have pre-event posts."
            ),
        )

    estimator_frame = unsupported_with_posts.merge(
        account_distance_quantiles.loc[:, ["did", "q80"]],
        on="did",
        how="inner",
    )
    valid_q80 = estimator_frame["q80"].dropna()
    if valid_q80.empty:
        return EarlyWindowRecommendation(
            needs_early_window=False,
            triggered_for_evaluation=True,
            accepted=False,
            early_window=None,
            candidate_early_window=None,
            continuous_q75_of_q80_short=None,
            n_positive_total=total_positives,
            n_unsupported_late=n_unsupported_late,
            unsupported_share=_maybe_float(unsupported_share),
            n_unsupported_with_posts=n_unsupported_with_posts,
            n_accounts_used_for_estimation=0,
            minimum_accounts_required=minimum_unsupported_with_posts_for_early_estimation,
            late_window=int(late_window),
            reason=(
                "Early window trigger was met, but q80_i_short is undefined for all "
                "unsupported-late positives with posts."
            ),
        )

    continuous_q75 = float(valid_q80.quantile(0.75))
    candidate_early_window = _choose_smallest_grid_window_at_or_above(
        continuous_q75,
        normalized_grid,
    )
    if candidate_early_window is None:
        raise RuntimeError("Early-window selection failed despite valid q80 estimates.")

    enough_accounts = (
        len(valid_q80) >= minimum_unsupported_with_posts_for_early_estimation
    )
    respects_scale = candidate_early_window <= (0.5 * float(late_window))
    accepted = bool(enough_accounts and respects_scale)

    if accepted:
        reason = (
            "Early window accepted because the unsupported-late trigger was met, "
            "the estimation sample is large enough, and the candidate window is at most half of the late window."
        )
    else:
        failures: list[str] = []
        if not enough_accounts:
            failures.append(
                f"only {len(valid_q80)} unsupported-late accounts with posts are available, "
                f"below the minimum of {minimum_unsupported_with_posts_for_early_estimation}"
            )
        if not respects_scale:
            failures.append(
                f"candidate_early_window={candidate_early_window} exceeds half of late_window={late_window}"
            )
        reason = "Early window rejected because " + " and ".join(failures) + "."

    return EarlyWindowRecommendation(
        needs_early_window=accepted,
        triggered_for_evaluation=True,
        accepted=accepted,
        early_window=int(candidate_early_window) if accepted else None,
        candidate_early_window=int(candidate_early_window),
        continuous_q75_of_q80_short=continuous_q75,
        n_positive_total=total_positives,
        n_unsupported_late=n_unsupported_late,
        unsupported_share=_maybe_float(unsupported_share),
        n_unsupported_with_posts=n_unsupported_with_posts,
        n_accounts_used_for_estimation=int(len(valid_q80)),
        minimum_accounts_required=minimum_unsupported_with_posts_for_early_estimation,
        late_window=int(late_window),
        reason=reason,
    )


def run_true_takedown_utility_window_analysis(
    fingerprints_parquet_path: str | Path = DEFAULT_FINGERPRINTS_PARQUET,
    labels_parquet_path: str | Path = DEFAULT_LABELS_PARQUET,
    posts_parquet_path: str | Path = DEFAULT_WINDOW_POSTS_PARQUET,
    observation_start: Any = DEFAULT_OBSERVATION_START,
    late_grid: Sequence[int] = DEFAULT_LATE_GRID,
    early_grid: Sequence[int] = DEFAULT_EARLY_GRID,
    unsupported_share_threshold: float = DEFAULT_UNSUPPORTED_SHARE_THRESHOLD,
    unsupported_count_threshold: int = DEFAULT_UNSUPPORTED_COUNT_THRESHOLD,
    minimum_unsupported_with_posts_for_early_estimation: int = (
        DEFAULT_MIN_UNSUPPORTED_WITH_POSTS_FOR_EARLY_ESTIMATION
    ),
    include_post_level_table: bool = True,
) -> TrueTakedownUtilityWindowAnalysisResult:
    """Run the full positive-only utility-window analysis driven by true takedown anchors."""

    observation_start_utc = to_utc_timestamp(observation_start)
    normalized_late_grid = _normalize_grid(late_grid, name="late_grid")
    normalized_early_grid = _normalize_grid(early_grid, name="early_grid")

    positive_anchor_result = build_true_takedown_positive_anchors(
        labels_parquet_path=labels_parquet_path,
        fingerprints_parquet_path=fingerprints_parquet_path,
        observation_start=observation_start_utc,
    )
    pre_event_post_result = extract_pre_event_posts_for_positives(
        posts_parquet_path=posts_parquet_path,
        positive_anchors=positive_anchor_result.positive_anchors,
        observation_start=observation_start_utc,
    )
    positive_anchors = attach_pre_event_post_diagnostics(
        positive_anchor_result.positive_anchors,
        pre_event_post_result.account_diagnostics,
    )
    account_distance_quantiles = compute_account_distance_quantiles(
        pre_event_post_result.pre_event_posts,
        quantiles=(0.80,),
    )
    late_grid_diagnostics = build_late_grid_diagnostics(
        positive_anchors=positive_anchors,
        pre_event_posts=pre_event_post_result.pre_event_posts,
        late_grid=normalized_late_grid,
    )
    late_recommendation = select_late_window(
        positive_anchors=positive_anchors,
        account_distance_quantiles=account_distance_quantiles,
        late_grid=normalized_late_grid,
    )
    early_recommendation = evaluate_early_window_need_and_selection(
        positive_anchors=positive_anchors,
        account_distance_quantiles=account_distance_quantiles,
        late_window=late_recommendation.late_window,
        early_grid=normalized_early_grid,
        unsupported_share_threshold=unsupported_share_threshold,
        unsupported_count_threshold=unsupported_count_threshold,
        minimum_unsupported_with_posts_for_early_estimation=(
            minimum_unsupported_with_posts_for_early_estimation
        ),
    )

    pre_event_posts = (
        pre_event_post_result.pre_event_posts
        if include_post_level_table
        else pd.DataFrame(
            columns=["did", "post_time", "post_uri", "takedown_time", "distance_days"]
        )
    )

    summary = {
        "observation_start_utc": observation_start_utc.isoformat(),
        "labels_parquet_path": str(Path(labels_parquet_path)),
        "fingerprints_parquet_path": str(Path(fingerprints_parquet_path)),
        "posts_parquet_path": str(Path(posts_parquet_path)),
        "labels_columns": dict(LABELS_COLUMN_MAP),
        "fingerprints_columns": dict(FINGERPRINTS_COLUMN_MAP),
        "posts_columns": dict(POSTS_COLUMN_MAP),
        "n_positive_total": int(len(positive_anchors)),
        "n_positive_with_pre_event_posts": int(positive_anchors["has_pre_event_posts"].sum()),
        "n_pre_event_posts_total": int(len(pre_event_post_result.pre_event_posts)),
        "late_recommendation": asdict(late_recommendation),
        "early_recommendation": asdict(early_recommendation),
    }

    return TrueTakedownUtilityWindowAnalysisResult(
        positive_anchors=positive_anchors,
        pre_event_posts=pre_event_posts,
        pre_event_post_account_diagnostics=pre_event_post_result.account_diagnostics,
        account_distance_quantiles=account_distance_quantiles,
        positive_consistency_diagnostics=(
            positive_anchor_result.positive_consistency_diagnostics
        ),
        fingerprint_mismatch_details=positive_anchor_result.fingerprint_mismatch_details,
        late_grid_diagnostics=late_grid_diagnostics,
        late_recommendation=late_recommendation,
        early_recommendation=early_recommendation,
        summary=summary,
    )


__all__ = [
    "DEFAULT_EARLY_GRID",
    "DEFAULT_FINGERPRINTS_PARQUET",
    "DEFAULT_LABELS_PARQUET",
    "DEFAULT_LATE_GRID",
    "DEFAULT_MIN_UNSUPPORTED_WITH_POSTS_FOR_EARLY_ESTIMATION",
    "DEFAULT_OBSERVATION_START",
    "DEFAULT_UNSUPPORTED_COUNT_THRESHOLD",
    "DEFAULT_UNSUPPORTED_SHARE_THRESHOLD",
    "DEFAULT_WINDOW_POSTS_PARQUET",
    "EarlyWindowRecommendation",
    "LateWindowRecommendation",
    "PositiveAnchorBuildResult",
    "PreEventPostsExtractionResult",
    "TrueTakedownUtilityWindowAnalysisResult",
    "attach_pre_event_post_diagnostics",
    "build_late_grid_diagnostics",
    "build_true_takedown_positive_anchors",
    "compute_account_distance_quantiles",
    "evaluate_early_window_need_and_selection",
    "extract_pre_event_posts_for_positives",
    "run_true_takedown_utility_window_analysis",
    "to_utc_series",
    "to_utc_timestamp",
]

if __name__ == "__main__":
    result = run_true_takedown_utility_window_analysis(
        include_post_level_table=False
    )

    print("\n=== Late Recommendation ===")
    print(result.late_recommendation)

    print("\n=== Early Recommendation ===")
    print(result.early_recommendation)

    print("\n=== Summary ===")
    print(result.summary)

    print("\n=== Late Grid Diagnostics ===")
    print(result.late_grid_diagnostics.to_string(index=False))

    print("\n=== Positive Consistency Diagnostics ===")
    print(result.positive_consistency_diagnostics.to_string(index=False))

    print("\n=== Positive Anchors (head) ===")
    print(result.positive_anchors.head(20).to_string(index=False))

    print("\n=== Pre-Event Post Account Diagnostics (head) ===")
    print(result.pre_event_post_account_diagnostics.head(20).to_string(index=False))

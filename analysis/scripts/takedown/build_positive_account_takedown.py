from pathlib import Path
from datetime import datetime
import csv
import gzip
import json
import pandas as pd

MANIFEST = Path("/share/storage/monade/rota/results/official_labeler_manifest_paths_mar2026.txt")

OUT_DIR = Path("/share/storage/monade/rota/results/account_takedown_positive_mar2026")
OUT_PARQUET = OUT_DIR / "account_takedown_positive_mar2026_SENSITIVE.parquet"
OUT_SUMMARY = OUT_DIR / "account_takedown_positive_mar2026_summary.json"

A = datetime.fromisoformat("2026-03-01T00:00:00+00:00")
B = datetime.fromisoformat("2026-04-01T00:00:00+00:00")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_ts(x: str) -> datetime:
    return datetime.fromisoformat(x.replace("Z", "+00:00"))

def is_account_level_uri(uri: str) -> bool:
    return uri.startswith("did:")

files = [
    Path(line.strip().split()[0])
    for line in MANIFEST.read_text().splitlines()
    if line.strip()
]

first_td = {}
n_td_events = {}

files_processed = 0
rows_seen = 0
rows_takedown = 0
rows_takedown_non_neg = 0
rows_account_level = 0
rows_in_window = 0
bad_rows = 0

for path in files:
    files_processed += 1
    print(f"[{files_processed}/{len(files)}] {path}", flush=True)

    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rows_seen += 1

                try:
                    if row.get("val") != "!takedown":
                        continue
                    rows_takedown += 1

                    if str(row.get("neg", "")).lower() == "true":
                        continue
                    rows_takedown_non_neg += 1

                    uri = row.get("uri", "")
                    if not is_account_level_uri(uri):
                        continue
                    rows_account_level += 1

                    ts = parse_ts(row["ts"])
                    if not (A <= ts < B):
                        continue
                    rows_in_window += 1

                    did = uri

                    if did not in first_td or ts < first_td[did]:
                        first_td[did] = ts

                    n_td_events[did] = n_td_events.get(did, 0) + 1

                except Exception:
                    bad_rows += 1

    except Exception as e:
        print(f"ERROR reading {path}: {e}", flush=True)

df = pd.DataFrame(
    {
        "did": list(first_td.keys()),
        "takedown_at": [
            first_td[did].isoformat().replace("+00:00", "Z")
            for did in first_td
        ],
        "n_takedown_events": [
            n_td_events[did]
            for did in first_td
        ],
    }
)

df = df.sort_values(["takedown_at", "did"]).reset_index(drop=True)
df.to_parquet(OUT_PARQUET, index=False)

summary = {
    "window_start": A.isoformat().replace("+00:00", "Z"),
    "window_end_exclusive": B.isoformat().replace("+00:00", "Z"),
    "manifest": str(MANIFEST),
    "files_processed": files_processed,
    "rows_seen": rows_seen,
    "rows_takedown": rows_takedown,
    "rows_takedown_non_neg": rows_takedown_non_neg,
    "rows_account_level_takedown": rows_account_level,
    "rows_account_level_takedown_in_window": rows_in_window,
    "unique_positive_accounts": len(df),
    "bad_rows": bad_rows,
    "output_parquet": str(OUT_PARQUET),
}

OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nDONE")
print("Positive accounts:", len(df))
print("Saved parquet:", OUT_PARQUET)
print("Saved summary:", OUT_SUMMARY)
print("\nSummary:")
print(json.dumps(summary, indent=2))
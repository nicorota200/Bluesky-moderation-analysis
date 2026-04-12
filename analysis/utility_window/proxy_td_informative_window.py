from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


DAY_NS = 86_400 * 1_000_000_000


def to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_utc_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def to_unix_ns(series: pd.Series) -> np.ndarray:
    return series.astype("int64").to_numpy()


def add_proxy_columns(
    accounts: pd.DataFrame,
    A,
    B,
    delta_days: int,
    created_col: str = "profile_created_at",
    last_activity_col: str = "last_activity",
) -> pd.DataFrame:
    A_utc = to_utc_timestamp(A)
    B_utc = to_utc_timestamp(B)
    delta = pd.Timedelta(days=int(delta_days))

    df = accounts.copy()
    if created_col in df.columns:
        df[created_col] = to_utc_series(df[created_col])
    df[last_activity_col] = to_utc_series(df[last_activity_col])

    df["proxy_td"] = df[last_activity_col] <= (B_utc - delta)
    df["duration_days"] = (
        (df[last_activity_col] - A_utc).dt.total_seconds() / 86_400.0
    )
    df = df.loc[df["duration_days"].notna()].copy()
    df = df.loc[df["duration_days"] >= 0].copy()
    return df


def compute_structural_support(
    accounts: pd.DataFrame,
    windows: Iterable[int],
    positives_mask: pd.Series | None = None,
    duration_col: str = "duration_days",
    proxy_col: str = "proxy_td",
) -> pd.DataFrame:
    windows = sorted({int(w) for w in windows if int(w) > 0})
    if positives_mask is None:
        positives = accounts.loc[accounts[proxy_col]].copy()
    else:
        positives = accounts.loc[positives_mask].copy()

    n_total = len(positives)
    rows = []
    for window in windows:
        n_supported = int((positives[duration_col] >= window).sum())
        rows.append(
            {
                "window_days": window,
                "n_positive_total": n_total,
                "n_positive_supported": n_supported,
                "support_share": (n_supported / n_total) if n_total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def assign_duration_regime(
    accounts: pd.DataFrame,
    early_cut_days: int,
    duration_col: str = "duration_days",
    proxy_col: str = "proxy_td",
) -> pd.DataFrame:
    df = accounts.copy()
    early_label = f"early_<=_{int(early_cut_days)}d"
    late_label = f"late_>_{int(early_cut_days)}d"
    df["duration_regime"] = "non_positive"
    early_mask = df[proxy_col] & (df[duration_col] <= early_cut_days)
    late_mask = df[proxy_col] & (df[duration_col] > early_cut_days)
    df.loc[early_mask, "duration_regime"] = early_label
    df.loc[late_mask, "duration_regime"] = late_label
    return df


def estimate_early_cut_days(
    accounts: pd.DataFrame,
    candidate_days: Iterable[int],
    min_early_share: float = 0.05,
    min_late_share: float = 0.20,
    duration_col: str = "duration_days",
    proxy_col: str = "proxy_td",
) -> tuple[int | None, pd.DataFrame]:
    positives = accounts.loc[accounts[proxy_col], duration_col].dropna().astype(float)
    positives = positives.loc[positives >= 0]
    if positives.empty:
        return None, pd.DataFrame()

    max_day = max(1, int(np.ceil(positives.max())))
    x = np.arange(1, max_day + 1, dtype=float)
    y = np.array([(positives >= day).mean() for day in x], dtype=float)

    X_one = np.column_stack([np.ones_like(x), x])
    beta_one, *_ = np.linalg.lstsq(X_one, y, rcond=None)
    y_one = X_one @ beta_one
    sse_one = float(np.sum((y - y_one) ** 2))

    rows = []
    for cut_day in sorted({int(c) for c in candidate_days if int(c) >= 1}):
        early_share = float((positives <= cut_day).mean())
        late_share = float((positives > cut_day).mean())
        valid = (early_share >= min_early_share) and (late_share >= min_late_share)

        left_mask = x <= cut_day
        right_mask = x > cut_day
        sse_two = np.nan
        if left_mask.sum() >= 2 and right_mask.sum() >= 2:
            X_left = np.column_stack([np.ones(left_mask.sum()), x[left_mask]])
            X_right = np.column_stack([np.ones(right_mask.sum()), x[right_mask]])
            beta_left, *_ = np.linalg.lstsq(X_left, y[left_mask], rcond=None)
            beta_right, *_ = np.linalg.lstsq(X_right, y[right_mask], rcond=None)
            y_left = X_left @ beta_left
            y_right = X_right @ beta_right
            sse_two = float(
                np.sum((y[left_mask] - y_left) ** 2)
                + np.sum((y[right_mask] - y_right) ** 2)
            )

        rows.append(
            {
                "cut_day": cut_day,
                "early_share": early_share,
                "late_share": late_share,
                "valid_candidate": valid,
                "sse_one_segment": sse_one,
                "sse_two_segment": sse_two,
                "sse_improvement": (
                    sse_one - sse_two if pd.notna(sse_two) else np.nan
                ),
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values("cut_day").reset_index(drop=True)
    valid_rows = diagnostics.loc[diagnostics["valid_candidate"]].copy()
    if valid_rows.empty:
        return None, diagnostics

    best_cut = int(
        valid_rows.sort_values(
            ["sse_improvement", "cut_day"], ascending=[False, True]
        ).iloc[0]["cut_day"]
    )
    diagnostics["selected"] = diagnostics["cut_day"].eq(best_cut)
    return best_cut, diagnostics


def build_positive_anchors(
    accounts: pd.DataFrame,
    positives_mask: pd.Series | None = None,
    max_positives: int | None = 5000,
    random_state: int = 42,
    did_col: str = "did_id",
    ref_col: str = "last_activity",
    duration_col: str = "duration_days",
) -> pd.DataFrame:
    if positives_mask is None:
        df = accounts.loc[accounts["proxy_td"]].copy()
    else:
        df = accounts.loc[positives_mask].copy()

    if max_positives is not None and len(df) > int(max_positives):
        df = df.sample(n=int(max_positives), random_state=random_state)

    anchors = df[[did_col, ref_col, duration_col]].copy()
    anchors[did_col] = anchors[did_col].astype(str)
    anchors = anchors.rename(columns={ref_col: "reference_time"})
    anchors["reference_time"] = to_utc_series(anchors["reference_time"])
    anchors = anchors.sort_values(did_col).reset_index(drop=True)
    return anchors


def prepare_signal_events(
    events: pd.DataFrame,
    did_col: str = "did_id",
    ts_col: str = "event_time",
    value_col: str | None = None,
    signal_name: str = "signal",
) -> pd.DataFrame:
    df = events.copy()
    df[did_col] = df[did_col].astype(str)
    df[ts_col] = to_utc_series(df[ts_col])
    if value_col is None or value_col not in df.columns:
        df["value"] = 1.0
    else:
        df["value"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    df = df.loc[df[did_col].notna() & df[ts_col].notna()].copy()
    return df[[did_col, ts_col, "value"]].rename(
        columns={did_col: "did_id", ts_col: "event_time"}
    )


def load_events_from_parquet_duckdb(
    parquet_path: str,
    did_ids: Iterable[str],
    did_col: str = "did_id",
    ts_col: str = "created_at",
    min_time=None,
    max_time=None,
    value_expr: str = "1.0",
) -> pd.DataFrame:
    if duckdb is None:  # pragma: no cover
        raise ImportError("duckdb non e disponibile nell ambiente corrente.")

    did_array = np.asarray(list(did_ids), dtype=object)
    did_values = pd.DataFrame({did_col: pd.Series(pd.unique(did_array), dtype=str)})
    min_utc = to_utc_timestamp(min_time) if min_time is not None else None
    max_utc = to_utc_timestamp(max_time) if max_time is not None else None

    where_parts = []
    if min_utc is not None:
        where_parts.append(
            f"p.{ts_col} >= TIMESTAMPTZ '{min_utc.isoformat(sep=' ')}'"
        )
    if max_utc is not None:
        where_parts.append(
            f"p.{ts_col} <= TIMESTAMPTZ '{max_utc.isoformat(sep=' ')}'"
        )
    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + " AND ".join(where_parts)

    con = duckdb.connect()
    con.register("did_filter", did_values)
    query = f"""
        SELECT
            CAST(p.{did_col} AS VARCHAR) AS did_id,
            p.{ts_col} AS event_time,
            {value_expr} AS value
        FROM parquet_scan('{parquet_path}') AS p
        INNER JOIN did_filter AS d
            ON CAST(p.{did_col} AS VARCHAR) = d.{did_col}
        {where_sql}
    """
    try:
        return con.execute(query).df()
    finally:
        con.close()


def filter_events_for_anchors(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    min_time=None,
    did_col: str = "did_id",
    ts_col: str = "event_time",
    ref_col: str = "reference_time",
) -> pd.DataFrame:
    events_df = events.copy()
    anchors_df = anchors[[did_col, ref_col]].copy()
    events_df[did_col] = events_df[did_col].astype(str)
    anchors_df[did_col] = anchors_df[did_col].astype(str)

    merged = events_df.merge(
        anchors_df,
        on=did_col,
        how="inner",
    )
    if min_time is not None:
        min_utc = to_utc_timestamp(min_time)
        merged = merged.loc[merged[ts_col] >= min_utc].copy()
    merged = merged.loc[merged[ts_col] < merged[ref_col]].copy()
    return merged


def exclude_one_event_at_reference(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    did_col: str = "did_id",
    ts_col: str = "event_time",
    ref_col: str = "reference_time",
) -> pd.DataFrame:
    events_df = events.copy()
    anchors_df = anchors[[did_col, ref_col]].copy()
    events_df[did_col] = events_df[did_col].astype(str)
    anchors_df[did_col] = anchors_df[did_col].astype(str)

    merged = events_df.merge(
        anchors_df,
        on=did_col,
        how="left",
    )
    merged["_drop_anchor_event"] = False

    at_reference = merged[ts_col].eq(merged[ref_col])
    if at_reference.any():
        anchor_hits = merged.loc[at_reference, [did_col, ts_col]].copy()
        anchor_hits["_rank"] = anchor_hits.groupby(did_col).cumcount()
        merged.loc[at_reference, "_rank"] = anchor_hits["_rank"].to_numpy()
        merged.loc[at_reference & merged["_rank"].eq(0), "_drop_anchor_event"] = True

    kept = merged.loc[~merged["_drop_anchor_event"], events_df.columns].copy()
    return kept.reset_index(drop=True)


def build_signal_index(
    events: pd.DataFrame,
    did_col: str = "did_id",
    ts_col: str = "event_time",
    value_col: str = "value",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if events.empty:
        return {}

    df = events.sort_values([did_col, ts_col]).copy()
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for did_id, group in df.groupby(did_col, sort=False):
        ts_ns = to_unix_ns(group[ts_col])
        values = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0).to_numpy()
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        index[str(did_id)] = (ts_ns, cumulative)
    return index


def window_sum_from_index(
    signal_index: dict[str, tuple[np.ndarray, np.ndarray]],
    did_id: str,
    start_time=None,
    end_time=None,
) -> float:
    record = signal_index.get(str(did_id))
    if record is None:
        return 0.0

    ts_ns, cumulative = record
    if len(ts_ns) == 0:
        return 0.0

    left = 0
    right = len(ts_ns)
    if start_time is not None:
        left = int(np.searchsorted(ts_ns, to_utc_timestamp(start_time).value, side="left"))
    if end_time is not None:
        right = int(np.searchsorted(ts_ns, to_utc_timestamp(end_time).value, side="left"))
    return float(cumulative[right] - cumulative[left])


def compute_window_features(
    anchors: pd.DataFrame,
    signal_events: dict[str, pd.DataFrame],
    A,
    windows: Iterable[int],
    did_col: str = "did_id",
    ref_col: str = "reference_time",
    duration_col: str = "duration_days",
) -> pd.DataFrame:
    A_utc = to_utc_timestamp(A)
    windows = sorted({int(w) for w in windows if int(w) > 0})

    features = anchors[[did_col, ref_col, duration_col]].copy().reset_index(drop=True)
    signal_indexes = {
        signal_name: build_signal_index(events)
        for signal_name, events in signal_events.items()
    }

    for signal_name, index in signal_indexes.items():
        total_col = f"{signal_name}__total_observed"
        total_values = []
        window_values = {window: [] for window in windows}

        for row in features.itertuples(index=False):
            did_value = getattr(row, did_col)
            ref_time = getattr(row, ref_col)
            duration_days = float(getattr(row, duration_col))

            total_values.append(
                window_sum_from_index(index, did_value, start_time=A_utc, end_time=ref_time)
            )

            for window in windows:
                if duration_days >= window:
                    start_time = ref_time - pd.Timedelta(days=window)
                    value = window_sum_from_index(
                        index,
                        did_value,
                        start_time=start_time,
                        end_time=ref_time,
                    )
                else:
                    value = np.nan
                window_values[window].append(value)

        features[total_col] = total_values
        for window in windows:
            features[f"{signal_name}__w{window}"] = window_values[window]

    return features


def build_relative_event_timeline(
    signal_events: dict[str, pd.DataFrame] | pd.DataFrame,
    anchors: pd.DataFrame,
    A,
    signal_name: str | None = None,
    did_col: str = "did_id",
    ts_col: str = "event_time",
    value_col: str = "value",
    ref_col: str = "reference_time",
) -> pd.DataFrame:
    A_utc = to_utc_timestamp(A)

    if isinstance(signal_events, dict):
        frames = []
        for name, df in signal_events.items():
            frames.append(
                build_relative_event_timeline(
                    df,
                    anchors=anchors,
                    A=A_utc,
                    signal_name=name,
                    did_col=did_col,
                    ts_col=ts_col,
                    value_col=value_col,
                    ref_col=ref_col,
                )
            )
        if not frames:
            return pd.DataFrame(
                columns=[
                    did_col,
                    "signal_name",
                    ts_col,
                    ref_col,
                    value_col,
                    "distance_days",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    filtered = filter_events_for_anchors(
        signal_events,
        anchors=anchors,
        min_time=A_utc,
        did_col=did_col,
        ts_col=ts_col,
        ref_col=ref_col,
    )
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                did_col,
                "signal_name",
                ts_col,
                ref_col,
                value_col,
                "distance_days",
            ]
        )

    filtered = filtered.copy()
    filtered["signal_name"] = signal_name or "signal"
    filtered["distance_days"] = (
        (filtered[ref_col] - filtered[ts_col]).dt.total_seconds() / 86_400.0
    )
    filtered = filtered.loc[filtered["distance_days"] > 0].copy()
    return filtered[
        [did_col, "signal_name", ts_col, ref_col, value_col, "distance_days"]
    ].reset_index(drop=True)


def weighted_quantile(
    values: np.ndarray,
    quantiles: Iterable[float],
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(list(quantiles), dtype=float)
    if values.size == 0:
        return np.full(len(quantiles), np.nan)

    if sample_weight is None:
        sample_weight = np.ones_like(values, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    valid = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight >= 0)
    values = values[valid]
    sample_weight = sample_weight[valid]
    if values.size == 0 or sample_weight.sum() == 0:
        return np.full(len(quantiles), np.nan)

    order = np.argsort(values)
    values = values[order]
    sample_weight = sample_weight[order]

    cumulative = np.cumsum(sample_weight)
    cumulative = cumulative / cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def compute_pre_event_global_curve(
    relative_events: pd.DataFrame,
    signal_name: str | None = None,
    distance_col: str = "distance_days",
    value_col: str = "value",
) -> pd.DataFrame:
    df = relative_events.copy()
    if signal_name is not None and "signal_name" in df.columns:
        df = df.loc[df["signal_name"] == signal_name].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "horizon_days",
                "cumulative_value",
                "cumulative_share",
                "incremental_share",
            ]
        )

    max_horizon = max(1, int(np.ceil(df[distance_col].max())))
    horizons = np.arange(1, max_horizon + 1, dtype=int)
    total_value = float(df[value_col].sum())
    cumulative_values = np.array(
        [df.loc[df[distance_col] <= horizon, value_col].sum() for horizon in horizons],
        dtype=float,
    )
    cumulative_share = (
        cumulative_values / total_value if total_value > 0 else np.full_like(cumulative_values, np.nan)
    )
    curve = pd.DataFrame(
        {
            "horizon_days": horizons,
            "cumulative_value": cumulative_values,
            "cumulative_share": cumulative_share,
        }
    )
    curve["incremental_share"] = curve["cumulative_share"].diff().fillna(
        curve["cumulative_share"]
    )
    return curve


def compute_pre_event_global_quantiles(
    relative_events: pd.DataFrame,
    quantiles: Iterable[float] = (0.50, 0.75, 0.80, 0.90, 0.95),
    signal_name: str | None = None,
    distance_col: str = "distance_days",
    value_col: str = "value",
) -> pd.DataFrame:
    df = relative_events.copy()
    if signal_name is not None and "signal_name" in df.columns:
        df = df.loc[df["signal_name"] == signal_name].copy()
    if df.empty:
        return pd.DataFrame(columns=["quantile", "distance_days"])

    q_values = weighted_quantile(
        df[distance_col].to_numpy(),
        quantiles=quantiles,
        sample_weight=df[value_col].to_numpy(),
    )
    return pd.DataFrame(
        {"quantile": list(quantiles), "distance_days": q_values}
    )


def compute_account_pre_event_quantiles(
    relative_events: pd.DataFrame,
    quantiles: Iterable[float] = (0.50, 0.75, 0.80, 0.90, 0.95),
    signal_name: str | None = None,
    did_col: str = "did_id",
    distance_col: str = "distance_days",
    value_col: str = "value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = relative_events.copy()
    if signal_name is not None and "signal_name" in df.columns:
        df = df.loc[df["signal_name"] == signal_name].copy()
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty

    rows = []
    for did_id, group in df.groupby(did_col):
        q_values = weighted_quantile(
            group[distance_col].to_numpy(),
            quantiles=quantiles,
            sample_weight=group[value_col].to_numpy(),
        )
        row = {did_col: did_id, "n_events": len(group), "total_value": group[value_col].sum()}
        for q, value in zip(quantiles, q_values):
            row[f"q{int(round(q * 100))}"] = value
        rows.append(row)

    account_quantiles = pd.DataFrame(rows)
    summary = account_quantiles.drop(columns=[did_col]).describe().T.reset_index()
    summary = summary.rename(columns={"index": "metric"})
    return account_quantiles, summary


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return np.nan
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def evaluate_pre_event_window_grid(
    features: pd.DataFrame,
    relative_events: pd.DataFrame,
    windows: Iterable[int],
    signal_name: str,
    did_col: str = "did_id",
    duration_col: str = "duration_days",
) -> pd.DataFrame:
    windows = sorted({int(w) for w in windows if int(w) > 0})
    total_col = f"{signal_name}__total_observed"
    global_curve = compute_pre_event_global_curve(relative_events, signal_name=signal_name)

    df_events = relative_events.copy()
    if "signal_name" in df_events.columns:
        df_events = df_events.loc[df_events["signal_name"] == signal_name].copy()
    total_global_value = float(df_events["value"].sum()) if not df_events.empty else 0.0

    rows = []
    for window in windows:
        window_col = f"{signal_name}__w{window}"
        valid_mask = features[window_col].notna()
        valid = features.loc[valid_mask, [did_col, duration_col, total_col, window_col]].copy()
        total_values = valid[total_col].to_numpy(dtype=float)
        window_values = valid[window_col].to_numpy(dtype=float)
        capture = np.full(len(valid), np.nan, dtype=float)
        positive_total_mask = total_values > 0
        capture[positive_total_mask] = (
            window_values[positive_total_mask] / total_values[positive_total_mask]
        )
        finite_capture = capture[np.isfinite(capture)]

        global_value = (
            float(df_events.loc[df_events["distance_days"] <= window, "value"].sum())
            if total_global_value > 0
            else np.nan
        )
        global_share = (
            global_value / total_global_value
            if total_global_value > 0 and pd.notna(global_value)
            else np.nan
        )
        previous_rows = [row for row in rows if pd.notna(row["global_cumulative_share"])]
        previous_share = previous_rows[-1]["global_cumulative_share"] if previous_rows else 0.0

        rows.append(
            {
                "signal_name": signal_name,
                "window_days": window,
                "n_accounts_total": len(features),
                "n_accounts_valid": int(valid_mask.sum()),
                "valid_share": float(valid_mask.mean()) if len(features) else np.nan,
                "n_accounts_with_total_signal": int((features[total_col] > 0).sum()),
                "active_share": (
                    float((valid[window_col] > 0).mean()) if len(valid) else np.nan
                ),
                "window_value_mean": float(valid[window_col].mean()) if len(valid) else np.nan,
                "window_value_median": float(valid[window_col].median()) if len(valid) else np.nan,
                "window_value_std": float(valid[window_col].std()) if len(valid) else np.nan,
                "window_value_p75": float(valid[window_col].quantile(0.75)) if len(valid) else np.nan,
                "window_value_p90": float(valid[window_col].quantile(0.90)) if len(valid) else np.nan,
                "window_value_max": float(valid[window_col].max()) if len(valid) else np.nan,
                "capture_share_mean": float(finite_capture.mean()) if len(finite_capture) else np.nan,
                "capture_share_median": float(np.median(finite_capture)) if len(finite_capture) else np.nan,
                "capture_share_p75": float(np.quantile(finite_capture, 0.75)) if len(finite_capture) else np.nan,
                "capture_share_p90": float(np.quantile(finite_capture, 0.90)) if len(finite_capture) else np.nan,
                "global_cumulative_value": global_value,
                "global_cumulative_share": global_share,
                "incremental_global_share": (
                    global_share - previous_share
                    if pd.notna(global_share) and pd.notna(previous_share)
                    else np.nan
                ),
                "corr_window_total_pearson": _safe_corr(valid[window_col], valid[total_col], method="pearson"),
                "corr_window_total_spearman": _safe_corr(valid[window_col], valid[total_col], method="spearman"),
                "corr_window_duration_pearson": _safe_corr(valid[window_col], valid[duration_col], method="pearson"),
                "corr_window_duration_spearman": _safe_corr(valid[window_col], valid[duration_col], method="spearman"),
            }
        )

    return pd.DataFrame(rows)


def find_ecdf_elbow(global_curve: pd.DataFrame) -> int | None:
    if global_curve.empty or len(global_curve) < 3:
        return None

    curve = global_curve[["horizon_days", "cumulative_share"]].dropna().copy()
    if len(curve) < 3:
        return None

    x = curve["horizon_days"].to_numpy(dtype=float)
    y = curve["cumulative_share"].to_numpy(dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)

    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return None

    distances = []
    for xi, yi in zip(x_norm, y_norm):
        point = np.array([xi, yi])
        distance = np.abs(np.cross(line, point - start)) / line_norm
        distances.append(distance)

    idx = int(np.argmax(distances))
    return int(curve.iloc[idx]["horizon_days"])


def recommend_pre_event_window(
    utility_curve: pd.DataFrame,
    structural_support: pd.DataFrame,
    global_curve: pd.DataFrame,
    signal_name: str,
    min_support_share: float = 0.25,
    min_supported_positives: int = 500,
    coverage_target: float = 0.80,
    recommendation_rule: str = "coverage",
) -> tuple[pd.DataFrame, dict]:
    if utility_curve.empty:
        empty = pd.DataFrame()
        return empty, {
            "signal_name": signal_name,
            "decision": "no_data",
            "recommended_window": None,
            "coverage_target_window": None,
            "elbow_day": None,
            "elbow_window": None,
        }

    decision_table = utility_curve.merge(
        structural_support,
        on="window_days",
        how="left",
    ).sort_values("window_days").reset_index(drop=True)

    decision_table["feasible"] = (
        decision_table["support_share"].ge(min_support_share)
        & decision_table["n_positive_supported"].ge(min_supported_positives)
    )
    decision_table["reaches_coverage_target"] = (
        decision_table["global_cumulative_share"].ge(coverage_target)
    )

    feasible = decision_table.loc[decision_table["feasible"]].copy()
    coverage_window = None
    if not feasible.empty:
        coverage_hits = feasible.loc[feasible["reaches_coverage_target"]].copy()
        if not coverage_hits.empty:
            coverage_window = int(coverage_hits.iloc[0]["window_days"])

    elbow_day = find_ecdf_elbow(global_curve)
    elbow_window = None
    if elbow_day is not None and not feasible.empty:
        candidates = feasible.loc[feasible["window_days"] >= elbow_day, "window_days"]
        if not candidates.empty:
            elbow_window = int(candidates.iloc[0])
        else:
            elbow_window = int(feasible["window_days"].max())

    if recommendation_rule == "elbow":
        recommended = elbow_window or coverage_window
        decision = "elbow_then_coverage"
    else:
        recommended = coverage_window or elbow_window
        decision = "coverage_then_elbow"

    if recommended is None and not feasible.empty:
        recommended = int(feasible["window_days"].max())
        decision = f"{decision}_fallback_largest_feasible"
    elif recommended is None:
        decision = f"{decision}_no_feasible_window"

    recommendation = {
        "signal_name": signal_name,
        "decision": decision,
        "recommended_window": recommended,
        "coverage_target": coverage_target,
        "coverage_target_window": coverage_window,
        "elbow_day": elbow_day,
        "elbow_window": elbow_window,
        "n_feasible_windows": int(feasible.shape[0]),
    }
    return decision_table, recommendation


def assess_intermediate_window_need(
    utility_curve: pd.DataFrame,
    early_window: int | None,
    late_window: int | None,
    positive_durations: pd.Series,
    early_cut_days: int,
    signal_name: str,
    min_gap_days: int = 5,
    min_middle_gain: float = 0.10,
    min_middle_duration_share: float = 0.10,
) -> dict:
    result = {
        "signal_name": signal_name,
        "early_window": early_window,
        "late_window": late_window,
        "gap_days": None,
        "early_cumulative_share": None,
        "late_cumulative_share": None,
        "middle_gain": None,
        "intermediate_candidate_window": None,
        "intermediate_duration_share": None,
        "needs_intermediate_window_analysis": False,
    }

    if early_window is None or late_window is None or utility_curve.empty:
        return result
    if late_window <= early_window:
        return result

    gap_days = int(late_window - early_window)
    result["gap_days"] = gap_days
    if gap_days < min_gap_days:
        return result

    curve = utility_curve.loc[utility_curve["signal_name"] == signal_name].copy()
    curve = curve.sort_values("window_days").reset_index(drop=True)
    early_row = curve.loc[curve["window_days"] == int(early_window)]
    late_row = curve.loc[curve["window_days"] == int(late_window)]
    if early_row.empty or late_row.empty:
        return result

    early_share = float(early_row.iloc[0]["global_cumulative_share"])
    late_share = float(late_row.iloc[0]["global_cumulative_share"])
    result["early_cumulative_share"] = early_share
    result["late_cumulative_share"] = late_share

    middle = curve.loc[
        (curve["window_days"] > int(early_window))
        & (curve["window_days"] < int(late_window))
    ].copy()
    if middle.empty:
        return result

    target_share = early_share + 0.5 * (late_share - early_share)
    candidate = middle.loc[middle["global_cumulative_share"] >= target_share].copy()
    if candidate.empty:
        candidate = middle.tail(1).copy()
    candidate_row = candidate.sort_values("window_days").iloc[0]
    candidate_window = int(candidate_row["window_days"])
    middle_gain = float(candidate_row["global_cumulative_share"] - early_share)
    duration_share = float(
        ((positive_durations > early_cut_days) & (positive_durations <= candidate_window)).mean()
    )

    result["middle_gain"] = middle_gain
    result["intermediate_candidate_window"] = candidate_window
    result["intermediate_duration_share"] = duration_share
    result["needs_intermediate_window_analysis"] = bool(
        (middle_gain >= min_middle_gain)
        and (duration_share >= min_middle_duration_share)
    )
    return result


def plot_utility_curve(utility_curve: pd.DataFrame, signal_name: str | None = None) -> None:
    df = utility_curve.copy()
    if signal_name is not None and "signal_name" in df.columns:
        df = df.loc[df["signal_name"] == signal_name].copy()
    if df.empty:
        print("Utility curve vuota.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(
        df["window_days"],
        df["global_cumulative_share"],
        marker="o",
        label="Quota cumulata di attivita catturata",
    )
    plt.bar(
        df["window_days"],
        df["incremental_global_share"],
        alpha=0.25,
        width=0.8,
        label="Quota incrementale aggiunta",
    )
    plt.xlabel("Window days")
    plt.ylabel("Share")
    plt.title("Utility curve sulle finestre candidate")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()


def plot_support_vs_utility(
    decision_table: pd.DataFrame,
    signal_name: str | None = None,
) -> None:
    df = decision_table.copy()
    if signal_name is not None and "signal_name" in df.columns:
        df = df.loc[df["signal_name"] == signal_name].copy()
    if df.empty:
        print("Decision table vuota.")
        return

    plt.figure(figsize=(8, 6))
    colors = np.where(df["feasible"], "tab:blue", "tab:gray")
    plt.scatter(df["support_share"], df["global_cumulative_share"], c=colors)
    for row in df.itertuples(index=False):
        plt.annotate(str(int(row.window_days)), (row.support_share, row.global_cumulative_share))
    plt.xlabel("Support share")
    plt.ylabel("Global cumulative share")
    plt.title("Supporto strutturale vs utility cumulata")
    plt.grid(alpha=0.2)
    plt.show()


def plot_pre_event_global_curve(
    global_curve: pd.DataFrame,
    signal_name: str | None = None,
    coverage_target: float | None = None,
    coverage_window: int | None = None,
    elbow_day: int | None = None,
) -> None:
    if global_curve.empty:
        print("Global curve vuota.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(
        global_curve["horizon_days"],
        global_curve["cumulative_share"],
        label="ECDF cumulata pre-evento",
    )
    if coverage_target is not None:
        plt.axhline(coverage_target, color="tab:green", linestyle="--", label=f"coverage target = {coverage_target:.0%}")
    if coverage_window is not None:
        plt.axvline(coverage_window, color="tab:orange", linestyle="--", label=f"coverage window = {coverage_window}d")
    if elbow_day is not None:
        plt.axvline(elbow_day, color="tab:red", linestyle=":", label=f"elbow = {elbow_day}d")
    plt.xlabel("Distanza massima da T (giorni)")
    plt.ylabel("Quota cumulata di attivita")
    title = "Curva cumulata delle attivita pre-evento"
    if signal_name is not None:
        title = f"{title} - {signal_name}"
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()


@dataclass
class AnalysisResult:
    anchors: pd.DataFrame
    features: pd.DataFrame
    structural_support: pd.DataFrame
    utility_curve: pd.DataFrame
    decision_table: pd.DataFrame
    recommendation: dict
    relative_events: pd.DataFrame
    global_curve: pd.DataFrame
    global_quantiles: pd.DataFrame
    account_quantiles: pd.DataFrame
    account_quantile_summary: pd.DataFrame


def run_informative_window_analysis(
    accounts: pd.DataFrame,
    signal_events: dict[str, pd.DataFrame],
    A,
    windows: Iterable[int],
    anchors: pd.DataFrame | None = None,
    max_positives: int = 5000,
    random_state: int = 42,
    positives_mask: pd.Series | None = None,
    min_support_share: float = 0.25,
    min_supported_positives: int = 500,
    coverage_target: float = 0.80,
    recommendation_rule: str = "coverage",
    primary_signal_name: str | None = None,
    pre_event_quantiles: Iterable[float] = (0.50, 0.75, 0.80, 0.90, 0.95),
) -> AnalysisResult:
    windows = sorted({int(w) for w in windows if int(w) > 0})
    if primary_signal_name is None:
        primary_signal_name = next(iter(signal_events.keys()))

    if anchors is None:
        anchors = build_positive_anchors(
            accounts,
            positives_mask=positives_mask,
            max_positives=max_positives,
            random_state=random_state,
        )

    features = compute_window_features(
        anchors=anchors,
        signal_events=signal_events,
        A=A,
        windows=windows,
    )
    structural_support = compute_structural_support(
        accounts=accounts,
        windows=windows,
        positives_mask=positives_mask,
    )
    relative_events = build_relative_event_timeline(
        signal_events=signal_events,
        anchors=anchors,
        A=A,
    )
    global_curve = compute_pre_event_global_curve(
        relative_events,
        signal_name=primary_signal_name,
    )
    global_quantiles = compute_pre_event_global_quantiles(
        relative_events,
        quantiles=pre_event_quantiles,
        signal_name=primary_signal_name,
    )
    account_quantiles, account_quantile_summary = compute_account_pre_event_quantiles(
        relative_events,
        quantiles=pre_event_quantiles,
        signal_name=primary_signal_name,
    )
    utility_curve = evaluate_pre_event_window_grid(
        features=features,
        relative_events=relative_events,
        windows=windows,
        signal_name=primary_signal_name,
    )
    decision_table, recommendation = recommend_pre_event_window(
        utility_curve=utility_curve,
        structural_support=structural_support,
        global_curve=global_curve,
        signal_name=primary_signal_name,
        min_support_share=min_support_share,
        min_supported_positives=min_supported_positives,
        coverage_target=coverage_target,
        recommendation_rule=recommendation_rule,
    )

    return AnalysisResult(
        anchors=anchors,
        features=features,
        structural_support=structural_support,
        utility_curve=utility_curve,
        decision_table=decision_table,
        recommendation=recommendation,
        relative_events=relative_events,
        global_curve=global_curve,
        global_quantiles=global_quantiles,
        account_quantiles=account_quantiles,
        account_quantile_summary=account_quantile_summary,
    )

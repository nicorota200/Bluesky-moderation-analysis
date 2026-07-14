from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import PercentFormatter


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT = REPORT_DIR.parent
FIG_DIR = REPORT_DIR / "figures"
DATA_DIR = ROOT / "datasets" / "balduf_anon_march_2026"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "negative": "#376795",
    "positive": "#c83f49",
    "accent": "#4b8f78",
    "muted": "#6b7280",
    "grid": "#d7dde5",
    "light": "#f5f7fa",
}
BUCKET_LABELS = ["0", "1", "2", "3-5", "6-10", "11-25", "26-50", "51-100", "101-250", "251-500", "501-1000", "1001+"]


def p(path: Path) -> str:
    return path.as_posix()


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()


def setup_ax(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, weight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def connect(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=COLORS["muted"])
    ax.add_patch(arrow)


def box(ax, xy, w, h, text, fc="#ffffff", ec="#334155"):
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)
    return rect


def con_blocks():
    pos = DATA_DIR / "positive_blocks_analysis_10d_mar2026_v3_anon.parquet"
    neg = DATA_DIR / "negative_blocks_analysis_10d_mar2026_v3_anon.parquet"
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW blocks AS
        SELECT 'positive' AS sample, * FROM read_parquet('{p(pos)}')
        UNION ALL
        SELECT 'negative' AS sample, * FROM read_parquet('{p(neg)}')
        """
    )
    return con


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    box(ax, (0.05, 0.58), 0.22, 0.22, "ATProto\nstream, profili,\npost, liste", fc="#eef6fb")
    box(ax, (0.37, 0.70), 0.25, 0.18, "Segnali comunitari\nblocchi, modlist,\nlabeler terzi", fc="#eef8f3")
    box(ax, (0.37, 0.42), 0.25, 0.18, "Moderazione ufficiale\nlabel e !takedown", fc="#fff1f2")
    box(ax, (0.74, 0.56), 0.22, 0.22, "Analisi pre-evento\n10 giorni prima\ndel takedown", fc="#f8fafc")
    connect(ax, (0.27, 0.69), (0.37, 0.79))
    connect(ax, (0.27, 0.69), (0.37, 0.51))
    connect(ax, (0.62, 0.79), (0.74, 0.67))
    connect(ax, (0.62, 0.51), (0.74, 0.67))
    ax.text(0.5, 0.16, "Domanda empirica: i segnali pubblici precedono e discriminano i takedown account-level?", ha="center", fontsize=11)
    savefig("fig01_architecture.pdf")


def fig_utility_window():
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_ylim(0, 1)
    ax.set_xlim(-11.2, 1.2)
    ax.axis("off")
    ax.hlines(0.5, -10, 0, color="#1f2937", linewidth=2)
    ax.scatter([-10, 0], [0.5, 0.5], s=90, color=[COLORS["accent"], COLORS["positive"]], zorder=3)
    ax.text(-10, 0.67, "$T_i - 10$ giorni", ha="center", fontsize=10)
    ax.text(0, 0.67, "$T_i$", ha="center", fontsize=10)
    ax.text(0, 0.30, "evento escluso", ha="center", fontsize=9, color=COLORS["muted"])
    ax.annotate("", xy=(-0.02, 0.23), xytext=(-9.98, 0.23), arrowprops=dict(arrowstyle="<->", color=COLORS["muted"], linewidth=1.4))
    ax.text(-5, 0.08, "feature osservate nella finestra $[T_i-10, T_i)$", ha="center", fontsize=11)
    savefig("fig02_utility_window.pdf")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    labels = [
        "Log ufficiali\n!takedown",
        "Anchor positivi\nprimo evento",
        "Pseudo-event time\nper negativi",
        "Feature 10 giorni\nblocchi e labels",
        "Bucket esposizione\npost x follow",
        "Analisi descrittiva\ne Random Forest",
    ]
    x0 = 0.02
    w = 0.145
    for i, lab in enumerate(labels):
        x = x0 + i * 0.165
        box(ax, (x, 0.38), w, 0.30, lab, fc="#f8fafc")
        if i < len(labels) - 1:
            connect(ax, (x + w, 0.53), (x + 0.165, 0.53))
    savefig("fig03_pipeline.pdf")


def fig_corner_distribution():
    df = pd.DataFrame(
        {
            "corner": ["00", "0p", "p0", "pp"],
            "negative": [0.3381, 0.1043, 0.1721, 0.3855],
            "positive": [0.6023, 0.1237, 0.0999, 0.1742],
        }
    )
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    setup_ax(ax, ylabel="Quota del campione")
    ax.bar(x - width / 2, df["negative"], width, label="Negativi", color=COLORS["negative"])
    ax.bar(x + width / 2, df["positive"], width, label="Positivi", color=COLORS["positive"])
    ax.set_xticks(x, df["corner"])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    savefig("fig04_corner_distribution.pdf")


def fig_lifetime():
    data = pd.DataFrame(
        {
            "corner": ["00", "0p", "p0", "pp"],
            "mean": [1.747585, 9.795944, 5.049990, 12.503611],
            "median": [0.000081, 1.847813, 1.012575, 1.935394],
            "n": [20205, 4579, 4378, 7573],
        }
    )
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    setup_ax(ax, ylabel="Giorni")
    ax.bar(x - width / 2, data["mean"], width, color=COLORS["positive"], label="Media")
    ax.bar(x + width / 2, data["median"], width, color=COLORS["accent"], label="Mediana")
    ax.set_xticks(x, [f"{c}\nN={n:,}".replace(",", ".") for c, n in zip(data["corner"], data["n"])])
    ax.legend(frameon=False)
    savefig("fig05_lifetime_corner.pdf")


def fig_blocks_general():
    metrics = pd.DataFrame(
        {
            "metric": ["Almeno un blocco", "Blocchi medi", "P95 blocchi"],
            "negative": [0.1809, 1.1073, 3],
            "positive": [0.2782, 3.2787, 9],
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (_, row) in zip(axes, metrics.iterrows()):
        setup_ax(ax, title=row["metric"])
        values = [row["negative"], row["positive"]]
        ax.bar(["Negativi", "Positivi"], values, color=[COLORS["negative"], COLORS["positive"]], width=0.55)
        if row["metric"] == "Almeno un blocco":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 0.35)
        else:
            ax.set_ylim(0, max(values) * 1.25)
    savefig("fig06_blocks_general.pdf")


def fig_post_bucket_blocks():
    con = con_blocks()
    df = con.execute(
        """
        WITH bucket_stats AS (
            SELECT
                sample,
                post_bucket_index,
                COUNT(*) AS n_obs,
                AVG(n_blocks_received_10d) AS mean_blocks_10d,
                AVG(has_block_10d) AS share_has_block_10d
            FROM blocks
            GROUP BY sample, post_bucket_index
        )
        SELECT
            post_bucket_index,
            MAX(CASE WHEN sample = 'positive' THEN n_obs END) AS n_pos,
            MAX(CASE WHEN sample = 'negative' THEN n_obs END) AS n_neg,
            MAX(CASE WHEN sample = 'positive' THEN mean_blocks_10d END) AS pos_mean,
            MAX(CASE WHEN sample = 'negative' THEN mean_blocks_10d END) AS neg_mean,
            MAX(CASE WHEN sample = 'positive' THEN share_has_block_10d END) AS pos_share,
            MAX(CASE WHEN sample = 'negative' THEN share_has_block_10d END) AS neg_share
        FROM bucket_stats
        GROUP BY post_bucket_index
        ORDER BY post_bucket_index
        """
    ).df()
    con.close()
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.36
    setup_ax(axes[0], ylabel="Quota con blocchi", title="Presenza di almeno un blocco")
    axes[0].bar(x - width / 2, df["neg_share"], width, color=COLORS["negative"], label="Negativi")
    axes[0].bar(x + width / 2, df["pos_share"], width, color=COLORS["positive"], label="Positivi")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xticks(x, BUCKET_LABELS[: len(df)], rotation=45, ha="right")
    axes[0].legend(frameon=False)
    setup_ax(axes[1], ylabel="Blocchi medi", title="Intensita' media")
    axes[1].bar(x - width / 2, df["neg_mean"], width, color=COLORS["negative"], label="Negativi")
    axes[1].bar(x + width / 2, df["pos_mean"], width, color=COLORS["positive"], label="Positivi")
    axes[1].set_xticks(x, BUCKET_LABELS[: len(df)], rotation=45, ha="right")
    savefig("fig07_post_bucket_blocks.pdf")


def fig_exposure_heatmap():
    con = con_blocks()
    df = con.execute(
        """
        WITH bucket_stats AS (
            SELECT
                sample,
                post_bucket_index,
                follow_10d_bucket_index AS follow_bucket_index,
                COUNT(*) AS n_obs,
                AVG(n_blocks_received_10d) AS mean_blocks
            FROM blocks
            GROUP BY sample, post_bucket_index, follow_10d_bucket_index
        ),
        wide AS (
            SELECT
                post_bucket_index,
                follow_bucket_index,
                MAX(CASE WHEN sample = 'positive' THEN n_obs END) AS n_pos,
                MAX(CASE WHEN sample = 'negative' THEN n_obs END) AS n_neg,
                MAX(CASE WHEN sample = 'positive' THEN mean_blocks END) AS pos_mean,
                MAX(CASE WHEN sample = 'negative' THEN mean_blocks END) AS neg_mean
            FROM bucket_stats
            GROUP BY post_bucket_index, follow_bucket_index
        )
        SELECT *,
               CASE WHEN n_pos >= 20 AND n_neg >= 200 AND neg_mean > 0
                    THEN pos_mean / neg_mean
                    ELSE NULL
               END AS ratio
        FROM wide
        """
    ).df()
    con.close()
    matrix = np.full((12, 12), np.nan)
    for _, r in df.iterrows():
        i = int(r["post_bucket_index"])
        j = int(r["follow_bucket_index"])
        if i < 12 and j < 12 and not pd.isna(r["ratio"]):
            matrix[i, j] = r["ratio"]
    fig, ax = plt.subplots(figsize=(8.3, 7))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=np.nanpercentile(matrix, 95))
    ax.set_xticks(np.arange(12), BUCKET_LABELS, rotation=45, ha="right")
    ax.set_yticks(np.arange(12), BUCKET_LABELS)
    ax.set_xlabel("Follow ricevuti nella finestra")
    ax.set_ylabel("Post nella finestra")
    ax.set_title("Rapporto blocchi medi positivi / negativi", fontsize=12, weight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Rapporto")
    savefig("fig08_exposure_heatmap.pdf")


def fig_daily_timing():
    con = con_blocks()
    parts = []
    for d in range(10):
        rel = -(d + 1)
        parts.append(f"SELECT sample, {rel} AS relative_day, n_blocks_day_{d} AS blocks_day, n_blocks_received_10d AS total_blocks FROM filtered")
    union = "\nUNION ALL\n".join(parts)
    df = con.execute(
        f"""
        WITH filtered AS (
            SELECT *
            FROM blocks
            WHERE NOT (post_bucket_index = 0 AND follow_10d_bucket_index = 0)
              AND n_blocks_received_10d > 0
        ),
        stacked AS (
            {union}
        )
        SELECT sample, relative_day,
               SUM(blocks_day)::DOUBLE / NULLIF(SUM(total_blocks), 0) AS share
        FROM stacked
        GROUP BY sample, relative_day
        ORDER BY sample, relative_day
        """
    ).df()
    con.close()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    setup_ax(ax, ylabel="Quota dei blocchi", xlabel="Giorni rispetto all'evento")
    for sample, color, label in [("negative", COLORS["negative"], "Negativi"), ("positive", COLORS["positive"], "Positivi")]:
        sub = df[df["sample"] == sample].sort_values("relative_day")
        ax.plot(sub["relative_day"], sub["share"], marker="o", linewidth=2.2, color=color, label=label)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(list(range(-10, 0)))
    ax.legend(frameon=False)
    savefig("fig09_daily_timing.pdf")


def fig_rf_progression():
    data = pd.DataFrame(
        {
            "setup": ["1", "2", "3", "4", "5", "6"],
            "f1": [0.371, 0.382, 0.377, 0.728, 0.563, 0.688],
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    setup_ax(ax, ylabel="CV F1")
    bars = ax.bar(data["setup"], data["f1"], color=["#9ca3af", "#9ca3af", "#9ca3af", "#6b7280", COLORS["accent"], COLORS["positive"]])
    ax.set_xlabel("Setup")
    ax.set_ylim(0, 0.82)
    for b, v in zip(bars, data["f1"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}", ha="center", fontsize=9)
    savefig("fig10_rf_progression.pdf")


def fig_corner_performance():
    data = pd.DataFrame(
        {
            "group": ["0p", "p0", "pp"],
            "accuracy": [0.720703, 0.737589, 0.757352],
            "roc_auc": [0.726013, 0.747718, 0.818529],
            "f1": [0.630344, 0.660031, 0.754088],
        }
    )
    x = np.arange(len(data))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    setup_ax(ax, ylabel="Metrica")
    ax.bar(x - width, data["accuracy"], width, label="Accuracy", color=COLORS["negative"])
    ax.bar(x, data["roc_auc"], width, label="ROC AUC", color=COLORS["accent"])
    ax.bar(x + width, data["f1"], width, label="F1 positivi", color=COLORS["positive"])
    ax.set_xticks(x, data["group"])
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    savefig("fig11_corner_performance.pdf")


def fig_feature_importance():
    data = pd.DataFrame(
        {
            "feature": [
                "day_0",
                "unique blockers 10d",
                "day_1",
                "day_8",
                "day_9",
                "day_2",
                "day_7",
                "day_6",
                "day_3",
                "day_5",
                "day_4",
            ],
            "importance": [0.442374, 0.277806, 0.066383, 0.035584, 0.035450, 0.033943, 0.028605, 0.022458, 0.022042, 0.019238, 0.016116],
        }
    ).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    setup_ax(ax, xlabel="Importanza impurity-based")
    ax.barh(data["feature"], data["importance"], color=COLORS["positive"])
    ax.set_xlim(0, 0.5)
    for i, v in enumerate(data["importance"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    savefig("fig12_feature_importance.pdf")


def fig_coverage_lift():
    data = pd.DataFrame(
        {
            "signal": [
                "Official post labels",
                "Skywatch account",
                "Blacksky account",
                "Profile account",
                "Skywatch post",
                "Blacksky post",
                "Modlist 10d",
            ],
            "coverage": [2.344, 2.944, 0.090, 1.913, 1.094, 0.004, 1.943],
            "lift": [np.nan, 4.24, 14.27, 12.36, 0.90, 0.50, np.nan],
        }
    )
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    setup_ax(ax1, ylabel="Coverage positivi (%)")
    ax1.bar(x, data["coverage"], color=COLORS["positive"], alpha=0.85, label="Coverage positivi")
    ax1.set_xticks(x, data["signal"], rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, data["lift"], color="#111827", marker="o", linewidth=1.8, label="Lift")
    ax2.set_ylabel("Lift")
    ax2.set_ylim(0, max(data["lift"].dropna()) * 1.25)
    for spine in ["top"]:
        ax2.spines[spine].set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    savefig("fig13_coverage_lift.pdf")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    fig_architecture()
    fig_utility_window()
    fig_pipeline()
    fig_corner_distribution()
    fig_lifetime()
    fig_blocks_general()
    fig_post_bucket_blocks()
    fig_exposure_heatmap()
    fig_daily_timing()
    fig_rf_progression()
    fig_corner_performance()
    fig_feature_importance()
    fig_coverage_lift()
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()

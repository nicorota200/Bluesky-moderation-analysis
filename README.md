# Bluesky-moderation-analysis
- Tutor: Carlo Alberto Bono
- University: Politecnico di Milano
- Study of moderation labels, account enforcement events, and blocklist dynamics in the Bluesky social network.

## Reproducibility scope

The repository is organized around the derived anonymized March 2026 parquet datasets. These datasets are required to rerun the main block-signal and Random Forest analyses, but they must not be published to GitHub.

Expected local data location, either one works:

- `balduf_anon_march_2026/`
- `datasets/balduf_anon_march_2026/`

The local workspace may use `datasets/balduf_anon_march_2026` as a symlink to `../balduf_anon_march_2026`. Both `datasets/` and `balduf_anon_march_2026/` are ignored by git.

## Main analysis notebooks

- `analysis/rf_and_shap_on_blocks.ipynb`: Random Forest and held-out validation from derived block-feature pools.
- `analysis/blocks_signal_v3.ipynb`: descriptive block-signal analysis from derived positive/negative block parquet files.
- `analysis/analisi_labels_march.ipynb`: official-label analysis retained for reference; still depends on old machine-specific label-log inputs.
- `analysis/analisi_blocklists.ipynb`: modlist analysis retained for reference; still depends on old machine-specific list/list-item inputs.

The held-out report artifacts are the canonical final outputs:

- `ROTA-NICOLA-heldout.pdf`
- `bluesky_presentazione_final_heldout.pptx`
- `bluesky_report_heldout/`
- `analysis/rf_main_heldout_metrics.json`

## Local legacy code

Old machine-specific scripts and exploratory notebooks are preserved locally under `_local/legacy/`. That folder is ignored by git and is not intended to be published.

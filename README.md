# Bluesky Moderation Analysis

Research project on whether public community signals on Bluesky, especially account blocks, contain information about later official account-level takedowns.

Tutor: Carlo Alberto Bono
University: Politecnico di Milano

## What To Open First

- `ROTA-NICOLA.pdf`: final report.
- `bluesky_presentazione_final.pptx`: final presentation.
- `analysis/rf_on_blocks.ipynb`: main Random Forest analysis and held-out validation.
- `analysis/blocks_signal_v3.ipynb`: descriptive analysis of block signals before takedown.

## Key Findings

- Blocks are the strongest public community signal among the signals analyzed.
- The signal is concentrated near the takedown event and is more informative for accounts with some observable exposure.
- Accounts with zero posts and zero incoming follows behave differently: many appear to be removed too quickly to accumulate public community signals.
- The main Random Forest model excludes the zero-exposure corner and uses 11 blocker-count features.
- Final held-out test metrics from the main model:
  - Accuracy: `0.7268`
  - Precision: `0.8409`
  - Recall: `0.5595`
  - F1: `0.6719`
  - ROC AUC: `0.7535`

## Repository Contents

```text
.
|-- ROTA-NICOLA.pdf
|-- bluesky_presentazione_final.pptx
|-- analysis/
|   |-- blocks_signal_v3.ipynb
|   |-- rf_on_blocks.ipynb
|   `-- scripts/analisi_predittiva/creazione_pool.ipynb
|-- requirements.txt
`-- README.md
```

The final report and presentation are the public deliverables. The notebooks are included to document and reproduce the main analysis from derived anonymized parquet datasets.

## Data Availability

The source datasets are not published in this repository. They are derived anonymized parquet files from the March 2026 Bluesky/ATProto analysis workspace.

Expected local data directory, either path works:

- `balduf_anon_march_2026/`
- `datasets/balduf_anon_march_2026/`

Both directories are ignored by git.

Minimum derived inputs for the public notebooks:

```text
positive_blocks_analysis_10d_mar2026_v3_anon.parquet
negative_blocks_analysis_10d_mar2026_v3_anon.parquet
raw_positive_pool_mar2026_created_anon.parquet
rf_1000_pool/
rf_10000_pool/
rf_10000_no00_01_pool/
rf_10000_balanced_4groups_pool/
rf_max_pool_no_00/
```

`analysis/scripts/analisi_predittiva/creazione_pool.ipynb` regenerates the RF pool folders from the positive and negative `v3` block-feature parquet files. It preserves the original sampling logic: build balanced positive/negative pools by exposure bucket, exclude low-utility corners for selected variants, and create the final max pool without the zero-exposure corner.

## Reproducibility

Create an environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Then place the derived parquet directory at one of the expected data paths and run:

1. `analysis/scripts/analisi_predittiva/creazione_pool.ipynb` if RF pool parquet files need to be regenerated.
2. `analysis/blocks_signal_v3.ipynb` for descriptive block-signal analysis.
3. `analysis/rf_on_blocks.ipynb` for Random Forest validation and held-out metrics.

The RF notebook writes `rf_main_heldout_metrics.json` as a generated local artifact. The JSON is ignored by git because it is reproducible from the notebook and derived data.

## Requirements

```txt
duckdb==1.5.4
graphviz
ipython
matplotlib==3.11.0
notebook==7.5.5
numpy==2.5.1
pandas==3.0.3
pyarrow==25.0.0
scikit-learn==1.9.0
scipy==1.18.0
seaborn==0.13.2
```

`pyarrow` is required for `pandas.read_parquet(...)`. `notebook` and `ipython` are included for running the notebooks interactively.

## Limitations

- The raw source data and derived anonymized parquet datasets are not included.
- The analysis is observational and correlational; it does not prove that blocks cause takedowns.
- Some supporting labeler and modlist explorations required non-public machine-specific inputs and are not part of the public reproducible repo.
- The final PDF is the polished report; notebooks are analysis artifacts rather than a production pipeline.

# Bluesky moderation report

Questo progetto contiene il report LaTeX:

**Segnali comunitari e moderazione ufficiale su Bluesky: analisi temporale e predittiva dei takedown account-level**

## Struttura

- `main.tex`: documento principale.
- `sections/`: sezioni del paper.
- `references.bib`: bibliografia autore-anno con backend `biber`.
- `figures/`: figure PDF vettoriali generate dallo script.
- `tables/audited_values.csv`: valori numerici auditati.
- `data_audit.md`: discrepanze e decisioni metodologiche.
- `scripts/generate_figures.py`: rigenerazione delle figure.

## Compilazione

Da questa cartella:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Il PDF e' impostato in stile paper a due colonne, con title/abstract a larghezza piena, figure in colonna e link interni/URL cliccabili.

Per generare direttamente il file finale con nome `ROTA-NICOLA.pdf`:

```bash
pdflatex -jobname=ROTA-NICOLA main.tex
biber ROTA-NICOLA
pdflatex -jobname=ROTA-NICOLA main.tex
pdflatex -jobname=ROTA-NICOLA main.tex
```

## Rigenerazione figure

Da radice repository:

```bash
.\.venv\Scripts\python.exe .\bluesky_report\scripts\generate_figures.py
```

Lo script usa i Parquet anonimizzati in `datasets/balduf_anon_march_2026` quando disponibili e valori auditati per le figure concettuali o per risultati riassuntivi gia' verificati nei notebook. Non usa screenshot delle celle Jupyter.

## Note metodologiche

- SHAP non e' stato usato. L'interpretazione del modello usa Random Forest feature importance nativa, basata sulla riduzione dell'impurita'.
- Le metriche predittive principali includono sia la cross-validation out-of-fold sul training set sia la valutazione separata sul test set held-out.
- Il report distingue account, eventi e osservazioni account-giorno; in particolare 44.395, 44.462 e 47.821 non sono valori intercambiabili.

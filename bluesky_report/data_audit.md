# Data audit

Questo audit documenta le scelte numeriche e metodologiche adottate nel report. La gerarchia usata e': codice e output dei notebook, script Python, `bluesky.md`, `MEMORIA.txt`, parte report di `bluesky_scaletta_bozze.md`, presentazione PowerPoint.

## Discrepanze e decisioni principali

- **SHAP**: il file `analysis/rf_and_shap_on_blocks.ipynb` contiene "shap" nel nome, ma l'analisi non usa SHAP, SHAP values o TreeExplainer. Il report usa solo Random Forest feature importance, cioe' impurity-based / Mean Decrease in Impurity. SHAP compare soltanto negli sviluppi futuri.
- **Numero di feature RF**: la cella finale del notebook stampa una lista storica di 28 feature, ma la matrice finale e' `X: (28650, 11)` e il codice usa il set a 11 feature basato su `n_unique_blockers_10d` e `n_unique_blockers_day_0`...`day_9`. Nel report l'analisi principale e' descritta come modello a 11 feature.
- **Cross-validation vs test set**: il notebook finale mantiene metriche out-of-fold sul training set e aggiunge una valutazione separata sul test set held-out del 30%. Il report distingue esplicitamente i due risultati.
- **44.395, 44.462, 47.821**: 44.395 e' il numero di account positivi distinti nel bucket 11--20 marzo; 44.462 e' il numero di eventi nello stesso bucket; 47.821 e' il numero di positivi nel dataset operativo 11--21 marzo. Il report mantiene i tre denominatori separati.
- **Corner 00 lifetime**: la media completa con `created_at` disponibile e' 1,7476 giorni su N=20.205; la media di circa 9,1 ore e' condizionata al sottoinsieme con lifetime inferiore a dieci giorni. Il report non afferma genericamente che la media del corner 00 sia nove ore.
- **Modlist**: per la finestra di dieci giorni il valore adottato e' 929 positivi in almeno una modlist, coverage 1,943%. Il valore 957, riferito a un criterio piu' ampio "ever before takedown", non e' usato come risultato 10-day.
- **Negativi e pseudo-event time**: la presentazione afferma un controllo sui dieci giorni successivi allo pseudo-evento, ma il codice e gli output disponibili non bastano a verificarlo. Il report adotta una formulazione prudente: negativi osservabili a marzo, creati prima dell'inizio mese, senza takedown nel periodo osservato e pseudo-event time distribuiti sugli stessi giorni dei positivi.
- **Manifest marzo**: i file della manifest associata a marzo possono contenere eventi con timestamp fuori mese. I conteggi finali usano il filtro sull'event timestamp.

## Valori non verificabili o verificati solo indirettamente

- Alcuni valori su labels di terze parti derivano da `bluesky.md` e dalla scaletta report; il notebook `analisi_labels_march.ipynb` e' stato ispezionato, ma non tutte le celle stampano tabelle finali complete. Nel testo si segnala che i negativi per alcuni labeler sono campionati.
- La procedura della utility window e' verificata dallo script `analysis/utility_window/true_takedown_utility_window.py`; non sono stati inventati output numerici non presenti nei materiali. Il report riporta solo il risultato operativo di 10 giorni.
- La causa sostantiva dei takedown non e' osservata. Il corner 00 e' interpretato come coerente con dinamiche anti-abuso account-level, non come prova di bot o spam per ogni account.

## File di supporto

La tabella CSV `tables/audited_values.csv` riporta affermazione, fonte, valori alternativi, scelta adottata e motivazione per i principali numeri del report.

- **RF held-out test**: nella versione held-out del report, il modello finale addestrato su tutto il training set ottiene sul test Accuracy 0.726818, Precision 0.840853, Recall 0.559460, F1 0.671884, ROC AUC 0.753478, CM [[3843,455],[1893,2404]]. Nessun overlap `did_anon` tra train/test o tra fit/validation fold CV.

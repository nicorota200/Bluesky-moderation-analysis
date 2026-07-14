# Bluesky — Bozze di lavoro
*Segnali comunitari e moderazione ufficiale su Bluesky/ATProto*
*Ultimo aggiornamento bozze: giugno 2026*

---

# PARTE 1 — SCALETTA REPORT

> Struttura accademica e discorsiva, seguendo `report_struttura.md`.  
> Sezione empirica organizzata attorno alla sequenza logica: utility window → gruppo 00 → analisi blocchi.  
> L'analisi più affidabile è l'ultima del notebook `rf_and_shap_on_blocks.ipynb` (pool max, no 00, 3 gruppi bilanciati).

---

## 1. Introduzione

**Motivazione e contesto**
- Bluesky è costruita su ATProto: la maggior parte del grafo sociale e dei metadati di moderazione è pubblicamente accessibile (blocchi, blocklist/modlist, labels, labeler).
- Questa apertura pone una domanda empirica concreta: i segnali di moderazione comunitaria *precedenti* un takedown hanno potere informativo rispetto alle successive decisioni di enforcement ufficiale?

**Domanda di ricerca principale**
- I segnali pubblicamente osservabili su Bluesky — blocchi ricevuti, inclusione in modlist, labels di labeler terzi — possono anticipare, correlare o predire il successivo takedown account-level (`!takedown`) da parte della piattaforma?

**Anteprima del risultato**
- Il segnale più robusto è quello dei blocchi ricevuti, in particolare per account con attività media o alta.
- Il gruppo di account senza post e senza follower (corner 00) richiede un trattamento separato: la sua dinamica non è spiegata dai segnali comunitari ma dalla risposta automatica di Bluesky agli account usa-e-getta.
- Labels di labeler terzi e modlist hanno coverage troppo bassa per essere predittivamente utili.

---

## 2. Domande di ricerca

- **Q1** — I blocchi ricevuti nei 10 giorni precedenti il takedown sono significativamente più elevati negli account sospesi rispetto ai controlli con lo stesso profilo di esposizione?
- **Q2** — Le modlist (blocklist comunitarie) nei 10 giorni precedenti sono un segnale anticipatorio rilevante?
- **Q3** — Le labels di labeler di terze parti (Skywatch Blue, Blacksky Moderation, Profile Labeller, altri) discriminano gli account che riceveranno un takedown?
- **Q4** — Chi sono gli account nel corner 00 (0 post, 0 follower)? Qual è la loro dinamica rispetto agli altri positivi?
- **Q5** — La label `!takedown` è un proxy affidabile per la sospensione ufficiale? Quanto sono frequenti revoche e takedown multipli?

---

## 3. Stato dell'arte

### 3.1 Moderazione nelle piattaforme decentralizzate
- Letteratura su moderazione centralizzata vs distribuita (Twitter/X, Facebook, YouTube).
- Emergere di protocolli aperti: ActivityPub, ATProto.
- Governance della moderazione in architetture federate.
- [TODO: Jhaver et al., Chandrasekharan et al., letteratura Mastodon/Fediverse]

### 3.2 Segnali comunitari come predittori di moderazione
- Studi su segnalazioni utenti, blocchi peer-to-peer, reputazione come segnale anticipatorio.
- **Lavoro Balduf** — stessi dataset ATProto; strutture dati da cui è costruito il dataset operativo del presente lavoro. Punto di riferimento metodologico principale.
- **Lavoro prof. Bono** — [TODO: descrivere contributo specifico al filone di ricerca]
- [TODO: altri riferimenti sul filone predittivo/correlazionale]

### 3.3 Spam, bot e account usa-e-getta
- Pattern degli account abusivi: breve durata, assenza di follower organici, attività di posting assente o ad alta frequenza.
- Report di trasparenza Bluesky 2023 e 2025: documentazione esplicita dell'uso di sistemi automatici contro spam, impersonation, ban evasion.
- Coerenza con il corner 00 osservato nel dataset (media vita ~9 ore, event_time − created_at < 10 giorni nella quasi totalità dei casi).
- [TODO: Ferrara et al., Varol et al., studi su bot detection]

---

## 4. Parte sperimentale

### 4.1 Dati

**Dataset fonte**
- Log del labeler ufficiale Bluesky per marzo 2026 (dataset Balduf, non anonimizzati, su macchina virtuale).
- Ispezione iniziale tramite `inspect_parquet_heads.py`; risultati salvati in `parquet_heads_INSPECTION_SENSITIVE.txt`.

**Struttura dei file**

| File | Contenuto | Dimensione |
|------|-----------|------------|
| `labeler_logs` (csv.gz) | Log eventi del labeler ufficiale (add/remove) | 4.189 file per marzo |
| `03-profiles.parquet` | Profili account (DID, created_at, ...) | 1.222.452 DID distinti |
| `03-lists.parquet` | Liste di moderazione | — |
| `03-list-items.parquet` | Account inclusi nelle liste (subject_id) | — |
| `03-list-blocks.parquet` | Adozione delle liste da utenti (did_id) | — |
| Post parquet (11–21/3) | Post nella finestra operativa | 36.946.883 righe |

**Nota sulla copertura temporale dei log**
- I file `labeler_log` nella manifest di marzo coprono anche chunk di gennaio e aprile; gli event timestamp includono anche eventi di febbraio (mese assente dalla manifest).
- Questo non impatta l'analisi di marzo ma spiega le discrepanze nei conteggi globali.

**Statistiche sui takedown di marzo 2026**

| Indicatore | Valore |
|-----------|--------|
| Finestra di osservazione | 2026-03-01 → 2026-04-01 (esclusivo) |
| Events `!takedown` non-neg (account-level, in window) | 159.161 |
| Events `!takedown` neg=true (revoche) | 3.705 (2,27%) |
| Account positivi unici (`!takedown` non-neg) | 158.803 |
| Account positivi con revoca successiva | 2.814 (1,77%) |
| Account con >1 takedown | 325 (0,20%) |
| Distanza media primo→secondo takedown | ~31,8h (max 688h) |
| Distanza media primo takedown→revoca | ~44,4h (max 622h) |
| Bad rows | 0 |

**Affidabilità del proxy `!takedown`**
- Frequenza revoche (1,77%) e takedown multipli (0,20%) basse: il proxy è robusto.
- Mantenere nel dataset positivi gli account con revoca successiva è conservativo; escluderli non modifica materialmente i risultati.

---

### 4.2 Metodologia

#### A. Utility window

La **utility window** è la finestra temporale pre-evento `[event_time − 10 giorni, event_time)` su cui vengono calcolate le exposure variables e i segnali per ogni account.

**Motivazione della scelta di 10 giorni**
- [TODO: inserire dettaglio sul calcolo della utility window — verrà fornito in futuro]
- La scelta bilancia copertura dei segnali (finestra abbastanza larga da catturare blocchi e labels) e disponibilità del dataset (limitato a marzo).
- La finestra impone un vincolo operativo: l'analisi è limitata ai positivi con `event_time` tra l'11 e il 21 marzo, per avere 10 giorni di storia osservabile senza sconfinare fuori dal dataset.
- Risultato: **44.462 positivi** operativi (bucket 11–20 marzo).

**Distribuzione takedown per periodo**

| Bucket | Account positivi | Events takedown |
|--------|-----------------|-----------------|
| 01–10 marzo | 48.485 | 48.565 |
| 11–20 marzo | 44.395 | 44.462 |
| 21–31 marzo | 65.923 | 66.134 |

#### B. Costruzione dataset positivi e negativi

**Positivi** — account con almeno un evento `!takedown` non-neg in marzo 2026, con `event_time` tra l'11 e il 21 marzo.

**Negativi** — account osservabili a marzo, creati prima del 2020-03-01, senza takedown nel periodo.

| Partenza | DID distinti |
|---------|--------------|
| DID distinti in `03-profiles.parquet` | 1.222.452 |
| Esclusi: creati prima del 2020 | 11 |
| Esclusi: creati da marzo 2026 in poi | 664.632 |
| Esclusi: positivi | 158.803 |
| **Raw negative pool** | **366.515** |

**Assegnazione pseudo-event_time ai negativi**
- Per ogni giorno nell'intervallo 11–21 marzo in cui si osservano positivi, i negativi ricevono lo stesso giorno come pseudo-`event_time`.
- Obiettivo: confrontare positivi e negativi sulla stessa finestra temporale precedente, riducendo il confounding calendare.

**Distribuzione giornaliera positivi**

| Giorno | Positivi |
|--------|---------|
| 2026-03-11 | 4.137 |
| 2026-03-12 | 4.002 |
| 2026-03-13 | 4.143 |
| 2026-03-14 | 4.478 |
| 2026-03-15 | 4.754 |
| 2026-03-16 | 4.987 |
| 2026-03-17 | 4.327 |
| 2026-03-18 | 3.755 |
| 2026-03-19 | 4.623 |
| 2026-03-20 | 5.189 |
| 2026-03-21 | 3.426 |

#### C. Exposure variables e stratificazione per bucket

**Exposure variables** calcolate per ogni account nella finestra di 10 giorni:
- `n_posts_10d` — numero di post pubblicati nei 10 giorni precedenti lo (pseudo-)event_time.
- `incoming_follows_from_march_start` — follow ricevuti dall'inizio di marzo fino allo (pseudo-)event_time.

**Schema di bucket lineari** (identico per entrambe le variabili):

| Bucket | Range |
|--------|-------|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3–5 |
| 4 | 6–10 |
| 5 | 11–25 |
| 6 | 26–50 |
| 7 | 51–100 |
| 8 | 101–250 |
| 9 | 251–500 |
| 10 | 501–1000 |
| 11 | 1001+ |

`exposure_bucket_index = post_bucket_index × 100 + follow_bucket_index`

*Scelta dei bucket lineari vs quantili*: i bucket lineari garantiscono somiglianza sostanziale (stesso livello di attività/visibilità) non solo numerica. Meno di 30 celle con numero di negativi molto basso (thin cells).

**Quattro macro-gruppi di esposizione (corner)**

| Corner | Definizione |
|--------|-------------|
| 00 | `post_bucket_index = 0` e `follow_10d_bucket_index = 0` |
| 0p | `post_bucket_index = 0` e `follow_10d_bucket_index > 0` |
| p0 | `post_bucket_index > 0` e `follow_10d_bucket_index = 0` |
| pp | `post_bucket_index > 0` e `follow_10d_bucket_index > 0` |

Distribuzione per campione:

| Corner | Negativi (share) | Positivi (share) |
|--------|-----------------|-----------------|
| 00 | 33,81% | 60,23% |
| 0p | 10,43% | 12,37% |
| p0 | 17,21% | 9,99% |
| pp | 38,55% | 17,42% |

Il corner 00 è *sovrarappresentato* nei positivi (60% vs 34% nei negativi).

#### D. Analisi predittiva: Random Forest e SHAP

**Modello**: `RandomForestClassifier` (scikit-learn), 500–1.000 alberi, `min_samples_leaf=5`, nessuna profondità massima, `class_weight=None` su pool bilanciati.

**Valutazione**: cross-validazione stratificata (StratifiedKFold, 5 o 10 fold), metriche: Accuracy, Precision, Recall, F1, ROC AUC.

**Features utilizzate**:
- Pool 1 (28 features): `n_blocks_received_10d`, `n_unique_blockers_10d`, `n_blocks_Xd` (giornalieri ×10), `n_unique_blockers_Xd` (giornalieri ×10), `n_blocks_0_1d/1_3d/3_7d/7_10d`, `log1p_blocks_received_10d`, `log1p_unique_blockers_10d`.
- Pool 2 (11 features, senza overlap): solo `n_unique_blockers_10d` + `n_unique_blockers_day_X` (×10).

**Progressione delle analisi** (dal notebook `rf_and_shap_on_blocks.ipynb`):
1. *Analisi 1* — pool 1.000+1.000, 28 features, pool completo.
2. *Analisi 2* — pool 1.000+1.000, 11 features (solo unique blockers), rimozione overlap.
3. *Analisi 3* — pool 10.000, stesso feature set dell'analisi 2, ampliamento campione.
4. *Analisi 4* — pool 10.000 no 00/01, esclusione corner 00 e bucket adiacenti.
5. *Analisi 5* — pool 10.000, 4 gruppi bilanciati (00/0p/p0/pp), analisi per-gruppo.
6. **Analisi 6 (principale)** — pool max (~28.650 obs, 20.055 training), no 00, 3 gruppi bilanciati (0p/p0/pp), 28 features, StratifiedKFold 10 fold. **Risultato più affidabile**.

---

### 4.3 Risultati

#### 4.3.1 Utility window e affidabilità del proxy

- Finestra di 10 giorni validata analiticamente. [TODO: aggiungere dettaglio del calcolo]
- `!takedown` è proxy affidabile della sospensione ufficiale (revoche 1,77%, takedown multipli 0,20%).
- Distribuzione giornaliera dei positivi stabile nel periodo operativo (range 3.426–5.189).

#### 4.3.2 Il corner 00: natura degli account usa-e-getta

Il corner 00 richiede trattamento separato prima di interpretare i risultati predittivi.

**Tempo di vita degli account positivi per macro-bucket (da `rf_and_shap_on_blocks.ipynb`, cell 3)**:

| Bucket | N | Media lifetime (giorni) | Mediana lifetime (giorni) |
|--------|---|------------------------|--------------------------|
| 00 | 20.205 | 1,75 | ~0 (0,000081) |
| 0p | 4.579 | 9,80 | 1,85 |
| p0 | 4.378 | 5,05 | 1,01 |
| pp | 7.573 | 12,50 | 1,94 |

*Lettura*: la mediana del corner 00 è prossima a zero (quasi la totalità ha vita < 1 giorno; media ~9 ore). Questi account vengono intercettati prima di accumulare qualsiasi segnale pubblico.

**Dati corner 00 nel dataset blocchi** (da `blocks_signal_v3.ipynb`, cell 13):

| Campione | N obs corner 00 | Block rate 10d | Mean blocks 10d | Mean unique blockers 10d |
|----------|-----------------|---------------|-----------------|-------------------------|
| Negative | 1.341.731 | 1,40% | 0,0175 | 0,0168 |
| Positive | 28.802 | 3,71% | 0,0867 | 0,0861 |

*Lettura*: i positivi in corner 00 hanno block rate doppio rispetto ai negativi 00, ma i valori assoluti sono bassissimi (mediana = 0 per entrambi). Il segnale blocchi nel corner 00 è statisticamente differente ma non operativamente utile.

**Spiegazione istituzionale** (da Bluesky Transparency Report 2025):
- I sistemi automatici Bluesky usano euristiche, pattern matching e modelli per intercettare spam patterns e known bot attack signatures; i segnali ad alta confidenza portano ad azione immediata.
- Le azioni account-level servono soprattutto contro impersonation, spam networks, coordinated manipulation e ban evasion — non contro il singolo contenuto.
- Coerenza con il dato empirico: event_time − created_at < 10 giorni nella quasi totalità del corner 00, media ~0,38 giorni (~9 ore).

#### 4.3.3 Segnale blocchi: analisi descrittiva (da `blocks_signal_v3.ipynb`)

**Statistiche generali** (all data, incluso corner 00):

| Campione | N obs | N account | Share con ≥1 blocco 10d | Mean blocchi 10d | P95 blocchi 10d | Mean unique blockers 10d |
|----------|-------|-----------|------------------------|-----------------|-----------------|-------------------------|
| Negative | 4.031.665 | 366.515 | 18,09% | 1,1073 | 3 | 1,0873 |
| Positive | 47.821 | 47.821 | 27,82% | 3,2787 | 9 | 3,2485 |

**Timing del segnale nei 10 giorni pre-evento** (escluso corner 00, da cell 9):

| Campione | Share blocchi negli ultimi 1d | Share blocchi negli ultimi 3d | Share blocchi nei giorni 3–10 |
|----------|------------------------------|------------------------------|------------------------------|
| Negative | 10,75% | 31,10% | 68,90% |
| Positive | 55,63% | 78,77% | 21,23% |

*Lettura critica*: nei positivi il 55% dei blocchi cade nell'ultimo giorno prima del takedown (vs 10% nei negativi). Nei giorni lontani dall'evento i positivi possono stare *sotto* ai negativi — perché sono sovrarappresentati nel corner 00, che per definizione accumula quasi zero blocchi. È un artefatto composizionale, non un paradosso.

**Segnale a parità di esposizione** — [TODO: inserire grafico sezione 3 del notebook `blocks_signal_v3.ipynb`]
- A parità di `exposure_bucket_index`, i positivi mostrano più blocchi dei negativi in quasi tutti i bucket.
- [TODO: inserire tabella o heatmap del ratio per bucket]

#### 4.3.4 Analisi predittiva — progressione delle Random Forest

**Analisi 1–3: pool 1.000–10.000, pool completo**

| Analisi | Pool | Features | CV F1 | Accuracy | Precision | Recall | AUC |
|---------|------|----------|-------|----------|-----------|--------|-----|
| 1 | 1.000+1.000 | 28 | 0,371 | 0,595 | 0,833 | 0,239 | — |
| 2 | 1.000+1.000 | 11 (unique blockers) | 0,382 | 0,596 | 0,815 | 0,250 | — |
| 3 | 10.000 | 11 | 0,377 | 0,600 | 0,854 | 0,242 | — |

*Lettura*: con il pool completo (incluso corner 00) il modello ha alta precision ma recall bassa. Riconosce bene i positivi che ha segnalato, ma ne segnala pochissimi. Il corner 00 (60% dei positivi) non è classificabile tramite il segnale blocchi.

**Feature importance (analisi 2, 11 features)**:
- `n_unique_blockers_day_0` — 44,2%
- `n_unique_blockers_10d` — 32,1%
- `n_unique_blockers_day_1` — 8,6%
- Giorni più lontani — quote residue decrescenti

**Analisi 4: pool 10.000, esclusione corner 00**

| Analisi | Pool | Esclusione | CV F1 | Accuracy | Precision | Recall |
|---------|------|-----------|-------|----------|-----------|--------|
| 4 | 10.000 no 00 | bucket 00 e 01 | **0,728** | **0,765** | **0,863** | **0,630** |

*Lettura*: escludendo il corner 00 il salto di performance è netto. F1 da 0,38 a 0,73. Recall da 0,24 a 0,63. La quasi totalità del rumore classificatorio era concentrata nel corner 00.

**Analisi 5: pool 10.000, 4 gruppi bilanciati (00/0p/p0/pp)**

Metriche per gruppo:

| Gruppo | N | Accuracy | Precision | Recall | F1 |
|--------|---|----------|-----------|--------|----|
| 00 | 3.521 | 0,515 | 0,859 | 0,031 | 0,061 |
| 0p | 3.513 | 0,706 | 0,901 | 0,450 | 0,600 |
| p0 | 3.461 | 0,725 | 0,930 | 0,489 | 0,641 |
| pp | 3.505 | 0,754 | 0,788 | 0,705 | 0,744 |

*Lettura*: gradiente chiaro. Il corner 00 è sostanzialmente inclassificabile (F1 = 0,06, recall 3%). Il gruppo pp (attività + follower) è il meglio classificato (F1 = 0,74). Il segnale blocchi cresce con l'attività dell'account.

**Analisi 6 — principale (pool max, no 00, 3 gruppi bilanciati)**

| Indicatore | Valore |
|-----------|--------|
| Dataset totale | 28.650 osservazioni |
| Training set | 20.055 (10.028 pos + 10.027 neg) |
| Test set | 8.595 (4.297 pos + 4.298 neg) |
| Gruppi nel training | 0p: 6.656 · p0: 6.768 · pp: 6.631 |
| CV F1 (10-fold) | **0,688** (std 0,013) |
| Accuracy | **0,739** |
| Precision | 0,853 |
| Recall | 0,576 |
| ROC AUC | **0,760** |

Metriche per macro-gruppo (analisi 6):

| Gruppo | N | Accuracy | ROC AUC |
|--------|---|----------|---------|
| 00 | 0 | — | — |
| 0p | 6.656 | 0,721 | 0,726 |
| p0 | 6.768 | 0,738 | 0,748 |
| pp | 6.631 | 0,757 | **0,819** |

Dettaglio per classe e gruppo:

| Gruppo | Classe | Precision | Recall | F1 |
|--------|--------|-----------|--------|----|
| 0p | 0 (neg) | 0,650 | 0,960 | 0,776 |
| 0p | 1 (pos) | 0,923 | 0,479 | 0,630 |
| p0 | 0 (neg) | 0,666 | 0,960 | 0,786 |
| p0 | 1 (pos) | 0,927 | 0,512 | 0,660 |
| pp | 0 (neg) | 0,743 | 0,779 | 0,761 |
| pp | 1 (pos) | 0,773 | 0,736 | 0,754 |

*Lettura finale*: il modello ha alta precision (85%) — quando segnala un account come positivo spesso ha ragione. Il recall rimane moderato (58%) — non riesce a catturare tutti i positivi. Il gruppo pp è il più classificabile (AUC 0,82). Il segnale blocchi è informativo e robusto, ma non completo: una parte rilevante dei takedown avviene su account che non hanno ancora accumulato blocchi visibili.

**Feature importance (analisi 6, 28 features)**:
- [TODO: inserire output feature importance / SHAP plot dall'analisi 6]
- Nelle analisi precedenti: `n_unique_blockers_day_0` è sempre la feature più importante (~44%), seguita da `n_unique_blockers_10d` (~28–32%). Il giorno immediatamente precedente al takedown ha il contributo maggiore.

#### 4.3.5 Modlist (blocklist comunitarie)

| Indicatore | Valore |
|-----------|--------|
| Account positivi totali (11–21 marzo) | 47.821 |
| Positivi in almeno una modlist nei 10d | 929 (1,94%) |
| Modlist distinte coinvolte | 73 |
| Media modlist per positivo incluso | 1,09 |
| Massimo modlist su un singolo positivo | 7 |

*Conclusione*: coverage < 2%. Non operativamente utile come segnale predittivo.

#### 4.3.6 Labels labeler ufficiale

Riepilogo livelli (marzo 2026):

| Livello | Tipi di label | Add | Remove | Oggetti distinti |
|---------|---------------|-----|--------|-----------------|
| Account-level | 16 | 405.546 | 46.829 | 298.528 |
| Post-level | 18 | 1.364.988 | 19.190 | 1.327.962 |

*Principale label account-level*: `needs-review` (243.983 add), poi `!takedown` (159.161), poi `spam` (1.005), `!suspend` (821).

Labels post-level nei positivi (finestra 10d):
- Solo **2,34%** dei positivi ha almeno una label su un post nei 10 giorni pre-takedown.
- Label più frequente: `sexual` (761 positivi, 1,59%), poi `porn` (499, 1,04%).
- Coverage troppo bassa per un ruolo anticipatorio.

#### 4.3.7 Labels labeler di terze parti

| Labeler | Livello | % pos labelizzati | % neg labelizzati | Lift |
|---------|---------|-------------------|-------------------|------|
| Skywatch Blue | Account | 2,944% | 0,695% | 4,24× |
| Blacksky Moderation | Account | 0,090% | 0,006% | 14,27× |
| Profile Labeller | Account | 1,913% | 0,155% | 12,36× |
| Skywatch Blue | Post | 1,094% | 1,209% | 0,90× |
| 10 altri labeler | — | 0% | 0% | 0× |

*Conclusione*: lift elevato per Blacksky e Profile Labeller, ma coverage quasi nulla. Nessun labeler di terze parti offre un segnale sufficientemente denso per uso predittivo.

---

## 5. Discussione e conclusioni

### 5.1 Sintesi

- **Blocchi ricevuti**: segnale robusto e replicabile. Performance predittive nette (F1 0,69–0,73, AUC 0,76–0,82) escludendo il corner 00. Il segnale si concentra nell'ultimo giorno pre-takedown (`n_unique_blockers_day_0` è sempre la feature più importante).
- **Corner 00**: non è "noise" ma una categoria distinta di account. Sono quasi tutti intercettati entro 9 ore dalla creazione da sistemi automatici, prima di accumulare segnali pubblici. Trattarli separatamente è metodologicamente necessario.
- **Labels e modlist**: coverage troppo bassa. Segnale assente o marginale nella finestra pre-takedown.

### 5.2 Natura del takedown e il corner 00

Il corner 00 rivela qualcosa di importante sulla natura del takedown account-level su Bluesky: non è solo enforcement di policy su contenuto, ma soprattutto risposta automatica a pattern di autenticità (spam, impersonation, ban evasion). I sistemi automatici agiscono prima che gli account acquisiscano visibilità pubblica — quindi prima che qualsiasi segnale comunitario possa formarsi. La moderazione comunitaria e quella istituzionale operano su *popolazioni diverse* di account.

### 5.3 Validità del proxy `!takedown`

Frequenza di revoca (1,77%) e takedown multipli (0,20%) basse. Il proxy è affidabile. La questione più rilevante non è l'affidabilità del proxy ma la composizione della popolazione: il mixing tra account usa-e-getta (corner 00) e account con attività reale (0p, p0, pp) è la principale fonte di eterogeneità.

### 5.4 Limiti

- **Finestra temporale**: solo marzo 2026. Possibili effetti stagionali o campagne spam specifiche non controllabili.
- **Confounding calendare**: mitigato con pseudo-event_time, non eliminabile.
- **Corner 00**: domina i positivi (60%) ed è quasi irrilevante per il segnale comunitario. Richiede trattamento separato.
- **Causalità**: analisi correlazionale. I blocchi precedono temporalmente il takedown, ma la direzione causale non è identificabile.
- **Coverage labeler**: un risultato null non esclude che il segnale esista ma non sia catturato dai labeler analizzati.
- **Calcolo utility window**: [TODO: aggiungere dettaglio sul metodo di calcolo quando disponibile]

### 5.5 Direzioni future

- Estensione temporale: più mesi per controllare la stagionalità.
- Feature engineering sul grafo dei blocchi: struttura, clustering, identità dei blockers.
- Analisi di sopravvivenza: modellare il tempo al takedown.
- Separazione corner 00 come filone autonomo sul rilevamento account usa-e-getta.
- Nuovi labeler con maggiore coverage nel tempo.

---
---

# PARTE 2 — SCALETTA PRESENTAZIONE

> Più pratica del report. Mostra non solo i risultati ma anche il lavoro svolto:  
> pipeline, dataset, scelte metodologiche, script, anonimizzazione, struttura repository.  
> Logica narrativa: motivazione → dati e pipeline → utility window → corner 00 → segnale blocchi → analisi RF → altri segnali → conclusioni.

---

## Slide 1 — Titolo

**Segnali comunitari e moderazione ufficiale su Bluesky / ATProto**
*Analisi predittiva e correlazionale — Marzo 2026*

- Autore / corso / data
- GitHub repository: [link]

---

## Slide 2 — Motivazione e domanda

**Bluesky è trasparente per design**
- ATProto rende pubblicamente osservabili blocchi, blocklist, labels, log dei labeler.
- Questa apertura permette una domanda empirica: i segnali comunitari *precedenti* un takedown hanno potere informativo?

**Domanda principale**
> I blocchi ricevuti, le modlist e le labels di labeler terzi possono anticipare o correlare con il successivo takedown account-level ufficiale (`!takedown`)?

---

## Slide 3 — Bluesky e ATProto: contesto minimo

- Architettura decentralizzata: Personal Data Servers (PDS), relay, AppView.
- Moderazione istituzionale: Moderation Service ufficiale emette labels account-level e post-level.
- Moderazione comunitaria: labeler terzi, modlist (blocklist adottabili da utenti), blocchi individuali.
- `!takedown` = label account-level emessa dal Moderation Service ufficiale = proxy per sospensione account.

---

## Slide 4 — Dataset e fonte

**Dataset Balduf (marzo 2026)**
- Log del labeler ufficiale (csv.gz) — 4.189 file per marzo.
- `03-profiles.parquet` — 1.222.452 DID.
- `03-lists.parquet`, `03-list-items.parquet`, `03-list-blocks.parquet`.
- Post parquet (11–21 marzo) — 36.946.883 righe.

**Accesso e sicurezza**
- Dati non anonimizzati su macchina virtuale (non spostabili).
- Per analisi locali: dataset derivati anonimizzati (`_anon.parquet`).
- [TODO: mostrare diagramma struttura repo con cartella `datasets/balduf_anon_march_2026/`]

---

## Slide 5 — Pipeline generale

```
Dati grezzi (VM)
      ↓
Ispezione parquet (inspect_parquet_heads.py)
      ↓
Costruzione positivi (positivi marzo, event_time 11–21/3)
      ↓
Costruzione negative pool (366.515 account)
      ↓
Exposure variables (n_posts_10d, incoming_follows)
      ↓
Bucketing lineare (post × follow → exposure_bucket_index)
      ↓
Costruzione dataset blocchi anonimizzati
      ↓
Analisi descrittiva (blocks_signal_v3.ipynb)
      ↓
Analisi predittiva — RF + SHAP (rf_and_shap_on_blocks.ipynb)
```

[TODO: inserire come diagramma visivo]

---

## Slide 6 — Takedown in marzo: numeri base

| Indicatore | Valore |
|-----------|--------|
| Events `!takedown` non-neg | 159.161 |
| Account positivi unici | 158.803 |
| Revoche (`neg=true`) | 2,27% degli eventi |
| Takedown multipli | 325 account (0,20%) |

**Il proxy `!takedown` è affidabile**: revoche e multipli sono marginali.

---

## Slide 7 — Utility window: perché 10 giorni?

**Il problema**
- Quanto deve essere lunga la finestra di osservazione pre-evento?
- Troppo corta: non cattura segnali; troppo lunga: contamina con segnali non pertinenti.

**Calcolo**
- [TODO: inserire dettaglio del metodo di calcolo della utility window quando disponibile]
- Risultato: **10 giorni** (`[event_time − 10d, event_time)`).

**Vincolo operativo**
- Con una utility window di 10 giorni, l'analisi è limitata ai positivi con `event_time` tra 11 e 21 marzo: **44.462 positivi**.
- Per ogni giorno i negativi ricevono lo stesso `event_time` come pseudo-evento (controllo del confounding calendare).

---

## Slide 8 — Struttura dei dataset costruiti

**Dataset positivi**
- `positive_blocks_analysis_10d_mar2026_v3_anon.parquet`
- Una riga per account: `did_anon`, `event_time`, `n_posts_10d`, `incoming_follows_10d`, `n_blocks_received_10d`, `n_unique_blockers_10d`, blocchi giornalieri (×10), bucket index.

**Dataset negativi**
- `negative_blocks_analysis_10d_mar2026_v3_anon.parquet`
- Stessa struttura, riga per osservazione giorno-per-giorno.

**Pool per Random Forest**
- `rf_1000_pool/`, `rf_10000_pool/`, `rf_10000_no00_01_pool/`, `rf_10000_balanced_4groups_pool/`, `rf_max_pool_no_00/`
- [TODO: mostrare struttura cartelle repository]

---

## Slide 9 — I quattro corner: chi sono i positivi?

**Distribuzione positivi e negativi per macro-bucket**

| Corner | Negativi | Positivi |
|--------|---------|---------|
| 00 (0 post, 0 follower) | 33,81% | **60,23%** |
| 0p (0 post, follower) | 10,43% | 12,37% |
| p0 (post, 0 follower) | 17,21% | 9,99% |
| pp (post, follower) | 38,55% | 17,42% |

**Il 60% dei positivi non ha né post né follower nella finestra.** Questo è il punto di partenza dell'analisi.

---

## Slide 10 — Il corner 00: account usa-e-getta

**Tempo di vita degli account positivi per bucket** (boxplot da `rf_and_shap_on_blocks.ipynb`)

| Bucket | Mediana lifetime |
|--------|----------------|
| 00 | ~0 giorni (~9 ore) |
| 0p | 1,85 giorni |
| p0 | 1,01 giorni |
| pp | 1,94 giorni |

**Interpretazione**
- Il corner 00 non rappresenta "utenti innocui": sono quasi esclusivamente account usa-e-getta.
- Vengono intercettati *prima* di accumulare qualsiasi segnale pubblico (post, follower, blocchi).
- Bluesky Transparency Report 2025: sistemi automatici agiscono su spam patterns e known bot signatures con azione immediata; le azioni account-level servono soprattutto contro spam networks e ban evasion.
- Il takedown qui è enforcement di autenticità, non di contenuto.

---

## Slide 11 — Segnale blocchi: overview descrittiva

**Dati generali** (dataset blocchi, incluso corner 00)

| | Negativi | Positivi |
|---|---------|---------|
| Share con ≥1 blocco 10d | 18,09% | 27,82% |
| Mean blocchi 10d | 1,11 | 3,28 |
| P95 blocchi 10d | 3 | 9 |

**Timing del segnale** (escluso corner 00)
- **55,6%** dei blocchi nei positivi cade nell'*ultimo giorno* pre-takedown (vs 10,8% nei negativi).
- Nei giorni lontani dall'evento, i positivi possono stare sotto i negativi: artefatto composizionale del corner 00.

[TODO: inserire grafico timing del segnale (pannello triplo, cell 9 di `blocks_signal_v3.ipynb`)]

---

## Slide 12 — Random Forest: evoluzione delle analisi

| Analisi | Pool | Esclusione | CV F1 |
|---------|------|-----------|-------|
| 1 | 1.000+1.000 | nessuna | 0,371 |
| 2 | 1.000+1.000 | nessuna (11 features) | 0,382 |
| 3 | 10.000 | nessuna | 0,377 |
| 4 | 10.000 | no 00 | **0,728** |
| 5 | 10.000 | nessuna (4 gruppi) | 0,563 |
| **6 (principale)** | **max (~28.650)** | **no 00** | **0,688** |

**Il salto da analisi 3 a analisi 4** (F1 0,38 → 0,73) dimostra che il corner 00 era il principale fattore di confusion nel modello.

---

## Slide 13 — Analisi principale (analisi 6): risultati

**Setup**
- Pool max, no 00, 3 gruppi bilanciati (0p/p0/pp), 28.650 osservazioni, StratifiedKFold 10 fold.

**Performance globale**

| Metrica | Valore |
|--------|--------|
| CV F1 | **0,688** |
| Accuracy | 0,739 |
| Precision | 0,853 |
| Recall | 0,576 |
| ROC AUC | **0,760** |

**Risultati per gruppo**

| Gruppo | Accuracy | ROC AUC | F1 (positivi) |
|--------|----------|---------|---------------|
| 0p | 0,721 | 0,726 | 0,630 |
| p0 | 0,738 | 0,748 | 0,660 |
| **pp** | **0,757** | **0,819** | **0,754** |

**Messaggio chiave**: gli account con sia post che follower sono i meglio classificabili tramite il segnale blocchi (AUC 0,82).

[TODO: inserire SHAP summary plot / feature importance chart dall'analisi 6]

---

## Slide 14 — Feature importance e interpretazione

**Feature più importanti** (analisi 2, 5, 6 — pattern consistente)
- `n_unique_blockers_day_0` — ~44% dell'importanza totale
- `n_unique_blockers_10d` — ~28–32%
- `n_unique_blockers_day_1` — ~8–9%
- Giorni più lontani — contributi decrescenti

**Lettura**: il giorno immediatamente precedente al takedown (`day_0`) è di gran lunga il più informativo. Il segnale blocchi è fortemente concentrato temporalmente vicino all'evento.

**Alta precision, recall moderata**
- Quando il modello dice "positivo", spesso ha ragione (precision 85%).
- Ma non riesce a catturare tutti i positivi (recall 58%): una parte riceve takedown senza aver accumulato blocchi visibili nella finestra.

---

## Slide 15 — Altri segnali: labels e modlist

**Labels labeler ufficiale (post-level, finestra 10d)**
- Solo **2,34%** dei positivi ha almeno una label su un post nei 10 giorni pre-takedown.
- Label più frequente: `sexual` (1,59%), `porn` (1,04%).
- Copertura troppo bassa per uso anticipatorio.

**Labels labeler di terze parti**

| Labeler | Lift | % pos | % neg |
|---------|------|-------|-------|
| Blacksky Mod. | 14,27× | 0,090% | 0,006% |
| Profile Labeller | 12,36× | 1,913% | 0,155% |
| Skywatch Blue (acc.) | 4,24× | 2,944% | 0,695% |
| 10 altri | 0× | 0% | 0% |

Lift elevato ma coverage quasi nulla. Non utilizzabile in modo predittivo.

**Modlist (blocklist)**
- **1,94%** dei positivi incluso in almeno una modlist nei 10d.
- 73 modlist distinte, prevalentemente account inclusi in liste generaliste.

**Conclusione comune**: segnale informativo potenzialmente presente, ma troppo rarefatto per uso sistematico.

---

## Slide 16 — Anonimizzazione e struttura repository

**Anonimizzazione**
- I DID reali vengono sostituiti con `did_anon` (hash deterministico) nei dataset locali.
- I dataset grezzi non anonimizzati restano solo su macchina virtuale.
- File derivati `_anon.parquet` contengono solo variabili aggregate (conteggi, bucket index) senza riferimenti a contenuto o identità.

**Struttura repository GitHub** [TODO: inserire screenshot / tree della repo]
```
Bluesky-moderation-analysis/
├── analysis/
│   ├── scripts/blocks/
│   │   └── blocks_signal_v3.ipynb
│   └── notebooks/
│       └── rf_and_shap_on_blocks.ipynb
├── datasets/
│   └── balduf_anon_march_2026/
│       ├── positive_blocks_analysis_10d_mar2026_v3_anon.parquet
│       ├── negative_blocks_analysis_10d_mar2026_v3_anon.parquet
│       ├── rf_1000_pool/
│       ├── rf_10000_pool/
│       ├── rf_10000_no00_01_pool/
│       ├── rf_10000_balanced_4groups_pool/
│       └── rf_max_pool_no_00/
└── scripts/
    └── [script di costruzione dataset]
```

---

## Slide 17 — Conclusioni

**Cosa abbiamo trovato**
1. **Utility window di 10 giorni** — scelta validata empiricamente.
2. **Il corner 00 è una categoria a parte** — account usa-e-getta intercettati automaticamente entro ore dalla creazione, prima di qualsiasi segnale comunitario.
3. **Il segnale blocchi è informativo** — per gli account con attività medio-alta (0p, p0, pp): F1 0,69, AUC 0,76 globalmente; AUC 0,82 nel gruppo pp.
4. **Il segnale è concentrato temporalmente** — `n_unique_blockers_day_0` è la feature più importante. Il giorno prima conta più dei 9 giorni precedenti messi insieme.
5. **Labels e modlist non scalano** — coverage troppo bassa per uso predittivo.

**Messaggio di fondo**
> La moderazione comunitaria su Bluesky *anticipa parzialmente* quella ufficiale — ma solo per la componente di account che ha avuto il tempo di accumulare segnali pubblici. La maggior parte dei takedown riguarda account usa-e-getta intercettati prima che il segnale comunitario possa formarsi.

---

## Slide 18 — Limiti e sviluppi futuri

**Limiti principali**
- Solo marzo 2026 — non generalizzabile a priori.
- Analisi correlazionale, non causale.
- Corner 00 richiede trattamento separato (ha dinamica propria non catturabile dai segnali analizzati).
- [TODO: aggiungere dettaglio calcolo utility window]

**Sviluppi futuri**
- Estensione temporale (più mesi).
- Analisi del grafo dei blocchi (chi blocca chi, clustering).
- Analisi di sopravvivenza (tempo al takedown come variabile dipendente).
- Filone autonomo sul rilevamento account usa-e-getta.

---

## Slide 19 — Materiali

- Repository GitHub: [link]
- Dataset anonimizzati: `datasets/balduf_anon_march_2026/`
- Notebook descrittivo: `blocks_signal_v3.ipynb`
- Notebook predittivo: `rf_and_shap_on_blocks.ipynb`
- [TODO: aggiungere riferimenti bibliografici definitivi: Balduf, prof. Bono, Jhaver, Ferrara, Bluesky Transparency Reports 2023/2025]

---

*— Fine bozze —*

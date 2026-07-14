Relazione del Progetto

---

UTILITy WINDOW

in questa parte ho calcolato quale fosse una utilizy window sufficiente e efficiente per studiare i takedown

risultato: 10 giorni

---

\-dati : balduf, non anominizzati, su macchina virtuale, completi di labels del labeler ufficiale. esclusivamente del mese di Marzo 2026

Prima attività:  
ho fatto una veloce ispezione dei file parquet disponibili sulla macchina.con inspect\_parquet\_heads.py, ho salvato i risultati in parquet\_heads\_INSPECTION\_SENSITIVE.txt, dove sono riportati per ogni dataset , le colonne , e un head.  
Questo mi serve per capire esattamente come i dati con cui lavoro sono esposti.

Seconda attività: (vedi notebook):

\-costruire i dataset che mi servono, iniziando da quello dei positivi, coloro che hanno ricevuto un \!takedown nel nostro periodo.

\- quanti sono i file [csv.gz](http://csv.gz) nei labelers\_log, per mese:  
2026-01 1014  
2026-03 4189  
2026-04 1247   
manca il mese di febbraio.

file immediatamente prima di marzo: 2026-01-18T22:27:22+00:00  1768775242.csv.gz

\-ma invece se guardo i ts dell’evento speifico, cè anche febbraio:  
Account-level non-neg \!takedown by event ts month:  
2026-01 58528  
2026-02 172431  
2026-03 159161  
2026-04 50

quindi nella manifest di marzo (che contiene anche un chunk di gennaio e uno di aprile), sono segnati anche eventi e log di: gen, feb  e aprile.

questo non influenza il mio operando però spiega alcuni dati.  
dallo script x viene fuori che:   
{  
  "window\_start": "2026-03-01T00:00:00Z",  
  "window\_end\_exclusive": "2026-04-01T00:00:00Z",  
  "rows\_takedown": 404477,  
  "rows\_takedown\_non\_neg": 397990,  
  "rows\_account\_level\_takedown": 390170,  
  "rows\_account\_level\_takedown\_in\_window": 159161,  
  "unique\_positive\_accounts": 158803,  
  "bad\_rows": 0,

l’analisi precedente spiega la discrepanza tra rows\_account\_level\_takedown\_in\_window (march) e rows\_account\_level\_takedown

ora ho tutti i positivi di marzo.

---

Analsi su takedown multiplo: vedi notebook

RISULTATI \- ACCOUNT CON PIÙ DI UN TAKEDOWN IN MARZO  
account con \>1 takedown: 325 (non neg)

Numero di takedown per account multiplo:  
mean     2.101538  
min      2.000000  
max     23.000000

Distanza tra primo e secondo takedown \- ore:  
mean     31.804620  
min       0.000002  
max     687.558989

Distanza tra primo e secondo takedown \- giorni:  
mean    1.325193e+00  
min     8.101852e-08  
max     2.864829e+01

nonostante sia solo 325 su 158803\. questo potrebbe aprire la questione dell’affidbilità del proxy label \!takedown, come sospensione ufficiale dell account.

Analsi preliminari sui positivi:

\- quanto è frequente la revoca di un takedown? (neg=true)

TAKEDOWN ACCOUNT-LEVEL \- MARZO 2026  
eventi \!takedown non-neg: 159161  
eventi \!takedown neg=true: 3705  
eventi totali \!takedown: 162866

Frequenza revoche su eventi:  
2.2749 %

Account:  
account con almeno un \!takedown non-neg: 158803  
account con almeno una revoca neg=true: 3661  
account positivi con revoca successiva: 2814

Frequenza account positivi con revoca successiva:  
1.772 %

\-se emessa con quanta distanza di tempo viene emessa la neg=true (mean, min, max)?

su account positivi con revoca successiva: 2814

Distanza primo takedown → prima revoca, in ore:  
mean     44.369661  
min       0.000802  
max     622.295698

Distanza primo takedown → prima revoca, in giorni:  
mean     1.848736  
min      0.000033  
max     25.928987

al momento nel mio dataset derivato : `account_takedown_positive_mar2026_SENSITIVE.parquet` , considero anche gli account ai quali successivamente è stato revocato il td (2814)

\-come sono distribuiti i takedown in marzo?  
posso considerare solo positivi con almeno 10 giorni di storia  
per quanto riguarderà i negativi invece:   
a causa della grandezza della mia finestra attuale (10 giorni) devo considerare account con una storia disponibile minima di 10 giorni, e anche sul bordo superiore, non posso sapere cosa succede dopo il limite della finestra.  
i negativi considerabili sono quindi quelli nei 10 giorni di mezzo del mese.

potrebbe essere comodo per il confronto confinare anche i postivi a 11 marzo \- 21 marzo.  
\-prima devo capire come sono distrbuiti i positivi e creare i dataset da usare.

 bucket\_10d  n\_positive\_accounts  n\_takedown\_events  
01-10 marzo                48485            48565.0  
11-20 marzo                44395            44462.0  
21-31 marzo                65923            66134.0

44462 è un buon numero, procedo.

per ogni giorno in cui osservo takedown positivi, campiono account negativi e assegno loro lo stesso giorno come pseudo-`event_time`.

In questo modo positivi e negativi vengono confrontati sulla stessa finestra temporale precedente, ad esempio `[event_time - 10 giorni, event_time)`, riducendo il rischio che le differenze dipendano dal momento calendario invece che dalla prossimità al takedown.

\-POSITIVI PER GIORNO

-        day  n\_positive\_accounts  
- 2026-03-11                 4137  
- 2026-03-12                 4002  
- 2026-03-13                 4143  
- 2026-03-14                 4478  
- 2026-03-15                 4754  
- 2026-03-16                 4987  
- 2026-03-17                 4327  
- 2026-03-18                 3755  
- 2026-03-19                 4623  
- 2026-03-20                 5189  
- 2026-03-21                 3426

2\. negative candidate pool   
Dato che il dataset operativo copre marzo 2026, considero negativi gli account osservabili a marzo, creati prima dell’inizio del mese, che non ricevono takedown nel periodo osservato.

- devo farlo su 03-profiles.parquet

 n\_rows  n\_distinct\_did   min\_created\_at          max\_created\_at  n\_created\_before\_march  
3414703         1222452 1-01-01 00:49:56 2056-03-18 01:15:32.232               1073751.0

SUMMARY  
DID distinti totali: 1.222.452

Account esclusi perché creati prima del 2020: 11

Account esclusi perché creati da marzo 2026 in poi: 664.632

Account creati tra il 2020 e prima di marzo 2026: 370.559

DID positivi esclusi: 158.803

Dimensione finale della raw negative pool: 366.515 account

nome file: /share/storage/monade/rota/results/account\_negative\_mar2026/raw\_negative\_pool\_mar2026.parquet

attenzione: per un errore “did\_id” diventa “did”

3\. calcolo file che tiene conto giorno per giorno dei negativi con una determinate finestra di 10 giorni

nome file: /share/storage/monade/rota/results/account\_negative\_mar2026/negative\_candidate\_observations\_by\_day\_10d\_mar2026.parquet

4\. calcolo exposure variables per positivi e negativi:  
\- n\_posts\_10d  
\- followers\_count

Lo script arricchisce le osservazioni negative giorno-per-giorno calcolando due variabili di esposizione: il numero di post pubblicati nei 10 giorni precedenti lo pseudo-`event_time` e il numero di follow ricevuti dall’inizio di marzo fino allo pseudo-`event_time`.

Il risultato è un nuovo dataset negativo con finestre temporali definite e misure di attività/visibilità utilizzabili per il successivo matching con i positivi.

SUMMARY

| Giorno | Osservazioni | DID distinti | Media `n_posts_10d` | Max `n_posts_10d` | Media `incoming_follows_from_march_start` | Max `incoming_follows_from_march_start` |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 2026-03-11 | 366.515 | 366.515 | 19,80 | 51.736 | 8,53 | 29.409 |
| 2026-03-12 | 366.515 | 366.515 | 19,96 | 51.947 | 9,32 | 29.443 |
| 2026-03-13 | 366.515 | 366.515 | 19,96 | 55.291 | 10,13 | 29.477 |
| 2026-03-14 | 366.515 | 366.515 | 20,01 | 103.306 | 10,90 | 29.524 |
| 2026-03-15 | 366.515 | 366.515 | 19,95 | 163.059 | 11,74 | 29.549 |
| 2026-03-16 | 366.515 | 366.515 | 19,85 | 199.535 | 12,61 | 29.589 |
| 2026-03-17 | 366.515 | 366.515 | 19,85 | 228.227 | 13,42 | 29.619 |
| 2026-03-18 | 366.515 | 366.515 | 20,04 | 267.623 | 14,27 | 30.053 |
| 2026-03-19 | 366.515 | 366.515 | 20,21 | 305.981 | 15,09 | 30.106 |
| 2026-03-20 | 366.515 | 366.515 | 20,33 | 340.142 | 15,86 | 30.141 |
| 2026-03-21 | 366.515 | 366.515 | 20,47 | 376.453 | 16,66 | 30.175 |

strano il comportamento di max posts

\-analysis

Metric    			Value

Direct rows    			36,946,883

Direct distinct posts    		36,946,321

Duplicated rows    		562

Min post time    		2026-03-11 01:00:00+01:00

Max post time    		2026-03-21 00:59:59.993000+01:00

va bene, sono insignificanti 552\.

5\. calcolo exposure dei positivi

| event\_day | n\_rows | n\_distinct\_dids | avg\_n\_posts\_10d | max\_n\_posts\_10d | avg\_incoming\_follows\_from\_march\_start | max\_incoming\_follows\_from\_march\_start |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 2026-03-11 | 4,137 | 4,137 | 10.307227 | 11,867 | 4.725647 | 1,630 |
| 2026-03-12 | 4,002 | 4,002 | 8.058721 | 6,183 | 14.188656 | 31,363 |
| 2026-03-13 | 4,143 | 4,143 | 12.026792 | 4,916 | 5.203958 | 1,620 |
| 2026-03-14 | 4,478 | 4,478 | 4.142028 | 1,903 | 3.402412 | 831 |
| 2026-03-15 | 4,754 | 4,754 | 7.807110 | 10,579 | 4.305006 | 1,581 |
| 2026-03-16 | 4,987 | 4,987 | 5.343894 | 7,868 | 3.490676 | 694 |
| 2026-03-17 | 4,327 | 4,327 | 9.909175 | 9,776 | 4.429859 | 1,153 |
| 2026-03-18 | 3,755 | 3,755 | 10.301731 | 6,177 | 6.279893 | 1,243 |
| 2026-03-19 | 4,623 | 4,623 | 7.762492 | 12,542 | 5.098637 | 1,375 |
| 2026-03-20 | 5,189 | 5,189 | 2.223164 | 700 | 4.118135 | 1,735 |
| 2026-03-21 | 3,426 | 3,426 | 2.451255 | 1,325 | 5.676007 | 3,230 |

6\. assegnazione bucket  
la nostra logica è:  
ogni positivo viene assegnato a uno strato di esposizione  
ogni negativo viene assegnato allo stesso schema di strati  
ogni positivo viene confrontato con tutti i negativi dello stesso giorno e dello stesso strato

bucket lineari siano più interpretabili dei quantili. I quantili garantiscono numerosità, ma non somiglianza sostanziale. I bucket lineari, invece, dicono chiaramente: questi account hanno più o meno lo stesso livello di attività/visibilità. 

posts e followers :   
0 \= 0  
1 \= 1  
2 \= 2  
3 \= 3–5  
4 \= 6–10  
5 \= 11–25  
6 \= 26–50  
7 \= 51–100  
8 \= 101–250  
9 \= 251–500  
10 \= 501–1000  
11 \= 1001+ 

dal controllo noto che ci sono \<30 celle con pochi negativi (thin) . valuterò dopo se considerarle e come trattarle

exposure\_bucket\_index \= post\_bucket\_index \* 100 \+ follow\_bucket\_index

---

ANALISI SUI BLOCCHI

1. con script ho creato due nuovi dataset contenendo informazioni sui blocchi  
2. li anonimizzo cosi da poterli mettere in locale. da qui poi poter fare i grafici  
3. plotto e mostro i dati e informazioni interessanti in notebook: 

vedi blocks\_signal\_v3.ipynb

—  
ANALISI PREDITTIVA E DI CORRELAZIONE (BLOCCHI)

Random Forest \= modello predittivo  
SHAP \= interpretazione del modello 

1 creo pool di positivi e negativi bilanciati. ne prendo 1000 e 1000 .  
con un analisi prendo tanti positivi di un bucket quanto sono in percentuale sul totale di positivi.  
2\. faccio altre prove con 1000, e togliendo coloro nel bucket 00, 01  
3 provo con modello da 10000 divisi in maniera equa tra 00, 0p, p0, pp e analizzo la differenza tra i gruppi .  
notebook: rf\_and\_shap\_on\_blocks.ipynb

in seguito analizzo meglio il fenomeno in analysis\\scripts\\blocks\\blocks\_signal\_v3.ipynb

\-faccio analisi su tempo di vita dei positivi in 00:  
quasi la totalità degli account nel corner 00 ha un event\_time \- created\_at \< 10 giorni,  
e la durata media di questi account è di circa 0.38 giorni (circa 9 ore).

la spiegazione più plausibile è che quel corner 00 non rappresenti “utenti innocui”, ma soprattutto account usa-e-getta intercettati molto presto dai sistemi anti-abuso di Bluesky.

Quello che è documentato ufficialmente punta proprio in quella direzione. Nel report di trasparenza pubblicato il 29 gennaio 2026, Bluesky dice che:

\-i sistemi automatici cercano in anticipo spam patterns e known bot attack signatures, usando euristiche, pattern matching e modelli; i segnali ad alta confidenza possono portare ad azione immediata (report 2025);  
\-le azioni account-level servono soprattutto contro impersonation, spam networks, coordinated manipulation e ban evasion, non contro il singolo post problematico;  
\-le feature abusate non sono solo i post: citano esplicitamente profiles, follows, starter packs e anche direct messages come superfici di moderazione/report;  
\-nel 2023 avevano già scritto di strumenti proattivi per rilevare slurs in handles, spam accounts, engagement farming e spam (Moderation Report 2023).

Quindi il risultato “event\_time \- created\_at \< 10 giorni, media 0.38 giorni” è coerente con questo scenario: account creati da poco, spesso monouso, che vengono fermati prima di accumulare segnali pubblici nei tuoi due indicatori (n\_posts\_10d e incoming\_follows\_10d).

C’è anche un altro pezzo importante: la documentazione Bluesky dice che le label possono colpire account, profile o content, e che i takedown account-level sono usati quando il problema è “il tipo di attore”, non il singolo contenuto (docs moderazione). Questo spiega bene perché il tuo dump di label, essendo in gran parte post/event-level, non riesca a “giustificare” molti takedown rapidi: è normale, perché i takedown più rapidi sembrano proprio appartenere al dominio authenticity / spam / impersonation, dove Bluesky dichiara di concentrare molta automazione e molta enforcement account-level.

l utlima analisi e la piu affidabile è l utlima fatta nel notebook rf\_and\_shap …

---

AGGIUNGO LABELS da labeler ufficiale

analizzo primi i csv\_logs di marzo per capire quali sono e che numeri hanno le labels.

 RIEPILOGO PER LIVELLO DELLA LABEL   
       label\_level  n\_label\_types  n\_label\_added  n\_label\_removed  n\_distinct\_labeled\_objects  
     account\_level             16         405546            46829                      298528  
other\_record\_level           11          22256            19796                       20840  
        post\_level                18        1364988            19190                     1327962

 ACCOUNT-LEVEL LABELS   
       label\_type  n\_label\_added  n\_label\_removed  n\_distinct\_labeled\_objects  
     needs-review         243983            42029                      137411  
        \!takedown           159161            3705                        158803  
             spam              1005                214                          942  
         \!suspend             821                797                            799  
           sexual               309                 35                             309  
            \!hide                104                 13                             104  
            \!warn                54                   6                              53  
             rude                30                  10                             30  
sexual-figurative          29                   3                              27  
    impersonation          24                 11                              24  
       intolerant               9                    1                              9  
             porn                 8                    2                             8   
        self-harm              5                    1                             5  
           nudity                2                    1                             2  
    graphic-media          1                    1                             1  
            rumor               1                    0                             1

 POST-LEVEL LABELS   
       label\_type  n\_label\_added  n\_label\_removed  n\_distinct\_labeled\_objects  
             porn             936908            10537                      936010  
           sexual            365140             6725                      329155  
           nudity             27520             1018                       27495  
sexual-figurative       12315               84                       12307  
             rude             8520              106                        8492  
    graphic-media       7745              410                        7732  
        \!takedown         2616              224                        2555  
       intolerant            2303               30                        2301  
        self-harm          1382               37                        1377  
           threat              476                4                         475  
             spam             31                4                          31  
        sensitive             12                2                          12  
            \!hide                9                4                           9  
       misleading            5                2                           5  
             gore                3                1                           3  
           corpse              1                0                           1  
    impersonation              1                0                           1  
     needs-review              1                2                           1

l’interesse principlae nel avere anche le labels era la possibilità di poter analizzare meglio i casi degli utenti bloccati con 0 post e 0 followers  
ma tutte queste labels sembrano legate ad un evento (post o altro)

needs-review:è una “flag account-level per revisione/moderazione”

\-analisi sulle label sui posts (che sembrano più numerose)

| Positivi totali | Con almeno una label | Senza alcuna label | % con almeno una label |
| ----- | ----- | ----- | ----- |
| 47.821 | 1.121 | 46.700 | 2,344% |

| Tipo di label | Positivi con label | % sui positivi totali | Post labelizzati distinti | Media post per positivo labelizzato | Mediana | Massimo post per un positivo |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| `porn` | 499 | 1,043% | 3.343 | 6,699 | 1 | 294 |
| `sexual` | 761 | 1,591% | 4.140 | 5,440 | 1 | 1.339 |
| `nudity` | 60 | 0,125% | 95 | 1,583 | 1 | 10 |
| `sexual-figurative` | 7 | 0,015% | 13 | 1,857 | 1 | 5 |
| `rude` | 23 | 0,048% | 56 | 2,435 | 1 | 12 |

| Numero di post labelizzati | Positivi | % tra i positivi con label `porn` |
| ----- | ----- | ----- |
| 1 | 259 | 51,904% |
| 2 | 57 | 11,423% |
| 3–5 | 85 | 17,034% |
| 6–10 | 54 | 10,822% |
| 11–25 | 24 | 4,810% |
| 26–50 | 6 | 1,202% |
| 51–100 | 9 | 1,804% |
| 101+ | 5 | 1,002% |
|  |  |  |

scarico e osservo altri labeler per vedere se ha senso proseguire con le label  
considera che i negativi sono 37566 (per non prendere la full scale che ci avrebbe messo troppo)

LABELERS  
                                       name         
\-Blacksky Moderation: eterogeneità funzionale   
\-Engagement Hacks Labeler:È tematicamente vicino a spam, manipolazione dell’engagement e comportamento artificiale.                 
\-AI Account Labeler: È account-oriented e potenzialmente utile per identificare account artificiali o automatizzati.   
\-Adult Content Filter and Moderation Service: È tematicamente vicino alle labels ufficiali più frequenti nel dataset, cioè `porn`, `sexual`, `nudity`                  
\-Skywatch Blue: È uno dei labeler esterni più noti e usati come servizio di moderazione alternativo/comunitario; 

nella finestra 10d 

| Livello | Labeler | % positivi labelizzati | % negativi labelizzati | Lift | Lettura |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Account-level | Skywatch Blue | 2,944% | 0,695% | 4,24× | Segnale debole ma reale |
| Account-level | Blacksky Moderation | 0,090% | 0,006% | 14,27× | Lift alto, coverage bassissima |
| Post-level | Skywatch Blue | 1,094% | 1,209% | 0,90× | Non discrimina i positivi |
| Post-level | Blacksky Moderation | 0,004% | 0,008% | 0,50× | Trascurabile |

quasi nullo

analizzo altri 10 labeler nei 10 giorni prima 

| Labeler | Livello | % positivi labelizzati | % negativi labelizzati | Lift | Lettura |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Profile Labeller | Account-level | 1,913% | 0,155% | 12,36× | Segnale raro ma molto più concentrato nei positivi |

  Tutti gli altri labeler hanno dato 0 labels negli ultimi 10 giorni (a nessun account)  
---

BLOCKLIST

`03-lists` identifica quali liste sono vere `modlist;`  
`03-list-items` identifica gli account inseriti in tali liste; (subject\_id )  
`03-list-blocks` misura invece l’adozione della lista da parte di utenti che la bloccano (did\_id ) 

In Bluesky il record **app.bsky.graph.list** può rappresentare liste di moderazione, liste di curation oppure liste di riferimento. Il campo purpose serve precisamente a distinguere questi casi. Il lexicon ufficiale definisce **app.bsky.graph.defs\#modlis**t come una lista di account sulla quale applicare un’azione aggregata di moderazione, cioè mute o block;

quali informazioni posso essermi utili per studiare le block list.  
\-sono interessanti le relazioni tra blocchi e blocklist, ma nel caso dei positivi i blocchi avvengono molto vicini al takedown.

\-relazioni tra blocklist e takwdown

prima facciamo delle analisi base come:

| Indicatore | Valore |
| ----- | ----- |
| Account positivi totali | 47.821 |
| Account positivi inclusi in almeno una modlist nei 10 giorni precedenti | 929 |
| Percentuale di positivi inclusi in almeno una modlist | 1,943% |
| DID positivi distinti inclusi in almeno una modlist | 929 |
| Inclusioni account–modlist osservate nei 10 giorni precedenti | 1.033 |
| Modlist distinte coinvolte | 73 |
| Numero medio di modlist per positivo, includendo gli account mai inclusi | 0,021 |
| Numero medio di modlist tra i soli positivi inclusi in almeno una lista | 1,090 |
| Mediana delle modlist tra i soli positivi inclusi | 1 |
| Numero massimo di modlist associate a un singolo positivo | 7 |

—----------------------


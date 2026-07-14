# DeepSight

Applicazione locale (Streamlit) per archiviare e cercare foto e video tramite
modelli di intelligenza artificiale: Qwen3-VL-Embedding-2B (ricerca semantica
testo/immagine, similarità visiva, tag automatici — servito in locale da un
sottoprocesso `llama-server`), MTCNN/FaceNet (riconoscimento volti e
raggruppamento in persone) e Whisper (trascrizione audio dei video).

## Avvio rapido

Gli script di installazione e avvio sono divisi per sistema operativo dentro
`scripts/`: `scripts\windows\` e `scripts/linux/`.

### Windows

1. **Installazione** (una tantum): doppio clic su `scripts\windows\install.bat`.
   Se Python non è presente nel sistema, lo scarica e installa automaticamente,
   poi crea l'ambiente virtuale `venv` e installa le dipendenze.
2. **Avvio applicazione**: doppio clic su `scripts\windows\run.bat`.
   Si apre automaticamente nel browser. Per chiudere: `Ctrl+C`.

> I `.bat` servono perché un `.ps1` con il doppio clic si aprirebbe nell'editor
> invece di eseguirsi: sono involucri che richiamano `install.ps1` / `run.ps1`
> nella stessa cartella.

> **Modelli Qwen (obbligatori, non scaricati dall'installatore)**: la cartella
> `models/qwen/` deve contenere a parte l'eseguibile `llama-server` (con le sue DLL),
> il modello `Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf` e la proiezione multimodale
> `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf`. Senza questi file l'app si avvia
> comunque, ma la prima ricerca testuale/per immagine, il primo caricamento o il
> calcolo dei tag automatici (tutto ciò che usa Qwen) falliscono con un errore
> esplicito — i volti (FaceNet) e il parlato (Whisper) non ne hanno bisogno.

### Linux / macOS

```bash
./scripts/linux/install.sh
./scripts/linux/run.sh
```

> A differenza di Windows, l'installatore **non** installa Python da solo (le
> distribuzioni differiscono troppo): se manca, indica il comando da usare.

> Versione Python consigliata: **3.13** (64-bit). Minimo richiesto: 3.11.

## Mappa dei file (per chi sviluppa)

Tutto il codice sorgente Python vive nella cartella **`src/`**:

| File / cartella        | Ruolo |
|------------------------|-------|
| `src/app.py`           | Interfaccia Streamlit e punto di ingresso dell'app. |
| `src/config.py`        | Impostazioni centrali (modelli, soglie) e percorsi delle cartelle dati. |
| `src/models.py`        | Caricamento e gestione dei modelli AI (Qwen3-VL via llama-server, FaceNet, Whisper) — oggetto `gestore`. |
| `src/qwen_client.py`   | Client HTTP per il sottoprocesso `llama-server`: lo avvia/ferma e gli chiede gli embedding di testo e immagini. |
| `src/processor.py`     | Pipeline di elaborazione a stadi (preparazione, embedding, volti, trascrizione) — coda persistente su SQLite. |
| `src/persone.py`       | Raggruppamento dei volti in persone (assegnazione greedy per centroide + re-cluster DBSCAN). |
| `src/database.py`      | Persistenza dei metadati su SQLite (e sincronizzazione con ChromaDB). |
| `src/chroma_store.py`  | Archivio vettoriale (ChromaDB) per la ricerca per similarità. |
| `src/calibra_soglie.py`| Script manuale: misura le distribuzioni reali dei coseni Qwen sull'archivio per tarare le soglie in `config.py`. |
| `src/test_models.py`   | Script diagnostico: verifica che i modelli AI (Qwen, FaceNet, Whisper) si carichino correttamente. |
| `src/test_qwen_client.py`, `src/test_database_v07.py`, `src/test_persone.py`, `src/test_ricerca_vettoriale.py` | Altri auto-test (`assert`-based, si eseguono direttamente). |
| `requirements.txt`     | Dipendenze Python di base. |
| `scripts/windows/`     | Installazione e avvio su Windows: `install.bat` / `run.bat` (involucri) e la logica PowerShell `install.ps1` / `run.ps1`. |
| `scripts/linux/`       | Installazione e avvio su Linux/macOS: `install.sh` / `run.sh`. |
| `models/qwen/`         | File del modello Qwen (`llama-server` + GGUF + mmproj) — non forniti dall'installatore, vedi sopra. |
| `data/`                | **Archivio gestito dall'app** (foto/video importati, frame, volti, anteprime, DB SQLite). |

> Nota: `config.py` e `chroma_store.py` risalgono di un livello rispetto a `src/`,
> quindi `data/` e `chroma_db/` restano nella cartella principale del progetto.

### Grafo delle dipendenze tra moduli

```
app.py  ──►  config, database, processor, models
processor.py  ──►  config, database, models, persone
persone.py  ──►  config, database
database.py  ──►  config  (─► chroma_store)
models.py  ──►  config, qwen_client
config.py  ──►  chroma_store
```

## Ricerca, coda ed elaborazione

### Ricerca

La scheda **🔍 Ricerca Avanzata** ha quattro modalità: testo (semantica, con Qwen),
similarità immagine (carica una foto di esempio), volti (carica un ritratto) e
parlato (cerca nelle trascrizioni Whisper dei video). Nella ricerca testuale si può
attivare un **prompt negativo** ("Cosa NON deve comparire"): il punteggio finale di
ogni risultato diventa `similarità_alla_query - λ · similarità_al_negativo` (λ è
`LAMBDA_PROMPT_NEGATIVO` in `config.py`), così i contenuti affini al prompt negativo
scendono in classifica invece di essere esclusi in modo netto.

### Coda di elaborazione

Caricare un file lo rende subito visibile in galleria (copia + registrazione nel
database sono istantanee); l'elaborazione AI vera e propria (embedding Qwen, volti,
trascrizione) avviene in background, a stadi, in un pannello nella barra laterale che
mostra quanti elementi restano, quello in corso e una stima del tempo residuo. La
coda è **persistita su SQLite**, non in memoria: se l'app viene chiusa a metà, alla
riapertura riprende esattamente da dove si era fermata, senza bisogno di reimportare
nulla. Il pannello permette di **mettere in pausa/riprendere** l'elaborazione (l'elemento
in corso viene comunque completato) e di **riprovare** gli elementi il cui stadio è
fallito.

### Persone (Face Database)

La scheda **👤 Persone** raggruppa automaticamente i volti rilevati nell'archivio: ogni
volto nuovo viene assegnato alla persona più simile o ne fonda una nuova. Aprendo una
persona si può assegnarle un **nome**, **unirla** con un'altra (i volti passano tutti
alla persona scelta) e vedere tutti i contenuti in cui compare. Il pulsante
"🔄 Ricalcola raggruppamenti" rilancia un re-clustering completo (DBSCAN) che corregge
eventuali fusioni sbagliate dell'assegnazione incrementale, preservando i nomi già
assegnati.

## Cartelle di servizio (generate automaticamente)

Queste cartelle non servono all'utente finale ma vengono ricreate/gestite dagli
strumenti. Restano **visibili** in Windows (nessun attributo *nascosto* viene più
impostato dagli script o dall'applicazione).

- `venv/` — ambiente virtuale Python (creato dall'installatore).
- `chroma_db/` — file del database vettoriale ChromaDB.
- `scripts/` — script di installazione/avvio, divisi in `windows/` e `linux/`.
- `__pycache__/` — cache dei bytecode Python (in `src/`).

## Nota sull'archivio `data/`

La cartella `data/` (in particolare `data/archive`) resta **visibile** ma è
**gestita dall'applicazione**: le foto e i video vanno importati dall'interno di
DeepSight, non copiati a mano nella cartella. I file aggiunti manualmente non
vengono indicizzati (niente tag, volti o embedding) e possono restare fuori
sincrono rispetto al database.

### Controllo integrità e file "intrusi"

Il controllo integrità vive nel pulsante con lo scudo **🛡️** della barra di
navigazione in alto, accanto a *Dashboard*, *Galleria* e *Ricerca Avanzata*. Il
pulsante riporta un badge con il numero di problemi rilevati (file intrusi +
record orfani) e passa dal colore neutro all'ambra/rosso quando ce n'è almeno
uno; un clic apre il pannello a comparsa con il dettaglio e le azioni correttive.
L'esito di ogni azione compare come notifica *toast*.

I casi rilevati sono due.

**File "intrusi"** — presenti in `data/archive` ma non registrati nel database
(tipicamente copiati a mano). Per gestirli:

- **📥 Importa e indicizza** — adotta i file: ne calcola l'hash, li rinomina con lo
  schema corretto e li elabora con la pipeline AI. I formati non supportati finiscono
  in quarantena.
- **🧹 Sposta in quarantena** — sposta i file in `data/quarantena/` senza indicizzarli,
  lasciando l'archivio pulito e allineato al database.

**Record orfani** — file registrati nel database ma spariti dal disco: compaiono
ancora in ricerca e galleria (con la sola miniatura). Con **🧹 Rimuovi dall'indice**
si eliminano i record insieme a vettori, miniature, frame e volti associati.

Se in `data/quarantena/` ci sono file, nel pannello appare anche un pulsante
**📂 Apri cartella quarantena (N)** che la apre nel gestore file del sistema.

Le stesse operazioni sono disponibili a livello di codice in `processor.py`
(`trova_file_intrusi`, `importa_file_intrusi`, `sposta_file_intrusi_in_quarantena`,
`trova_record_orfani`, `rimuovi_record_orfani`).

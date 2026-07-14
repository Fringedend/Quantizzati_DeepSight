# DeepSight v0.7 — Qwen3-VL Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CLIP+EasyOCR with Qwen3-VL-Embedding-2B (GGUF via llama-server), add negative-prompt search, a Face Database ("Persone") over the existing FaceNet embeddings, and a pausable per-stage background queue.

**Architecture:** Evolve in place. SQLite stays the source of truth, ChromaDB the ANN index. A vendored, already-tested `LlamaServer` HTTP client (from `model testing/benchmark/embed.py`) talks to a `llama-server.exe` subprocess managed lazily by `models.gestore`. The monolithic AI pipeline splits into per-stage columns (`stato_embedding`, `stato_volti`, `stato_trascrizione`) processed by one daemon worker thread that polls SQLite — crash-safe, pausable via a settings row.

**Tech Stack:** Python 3.11+, Streamlit, SQLite, ChromaDB, llama.cpp (llama-server, CPU), facenet-pytorch (unchanged), openai-whisper (unchanged), scikit-learn DBSCAN (already installed).

## Global Constraints

- All code identifiers, comments, and docs in **Italian** (repo rule, CLAUDE.md). Exception: the two vendored files copied verbatim from the tested benchmark (`src/qwen_client.py`) keep their English internals; their *wrapper* (`GestoreQwen`) is Italian.
- No new Python dependencies. The Qwen client uses only stdlib (`urllib`, `subprocess`, `json`) + numpy + PIL.
- Tests are `assert`-based scripts with `if __name__ == "__main__"` run directly with the venv python — **no pytest** (repo convention).
- Repo paths: project root is `DeepSight/`. All source in `src/`; scripts in `scripts/windows|linux/`. Flat imports (`import config`), scripts run with `src/` as cwd.
- Model runtime dir: `models/qwen/` at project root (git-ignored), containing `llama-server.exe` + its DLLs, the 2B GGUF, and the mmproj GGUF.
- Embedding dimension: **2048** (Qwen3-VL-Embedding-2B). FaceNet faces stay 512-d.
- llama-server launch recipe (verified in benchmark): `--embedding --pooling last --embd-normalize 2 -ngl 0` with env `LLAMA_MEDIA_MARKER=<__media__>`.
- The `media_frames.clip_embedding` column keeps its name (holds Qwen vectors now) — renaming would touch every query for zero benefit. A comment marks this.
- `processed` semantics change: `1` = "prepared" (thumbnail+EXIF+frames extracted, visible in gallery), not "fully AI-processed". AI completion is per-stage.

## Source files to copy (input artifacts)

| From (`model testing/`) | To (`DeepSight/`) |
|---|---|
| `Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf` | `models/qwen/` |
| `benchmark/runtime/mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` | `models/qwen/` |
| `benchmark/runtime/llama-server.exe` + all `*.dll` in that folder | `models/qwen/` |
| `benchmark/embed.py` + `benchmark/preprocess.py` | merged into `src/qwen_client.py` (Task 2) |

---

### Task 1: Repo setup — git init, model files, config entries

**Files:**
- Create: `.gitignore`
- Create: `models/qwen/` (binary files, not committed)
- Modify: `src/config.py`

**Interfaces:**
- Produces (used by every later task): `config.PERCORSO_LLAMA_SERVER`, `config.PERCORSO_MODELLO_QWEN`, `config.PERCORSO_MMPROJ_QWEN`, `config.QWEN_HOST`, `config.QWEN_PORT`, `config.QWEN_THREADS`, `config.DIM_EMBEDDING_QWEN = 2048`, `config.ISTRUZIONE_RICERCA`, `config.LAMBDA_PROMPT_NEGATIVO`, `config.SCALA_LOGIT_TAG`.

- [ ] **Step 1: Initialize git** (repo is not under version control yet)

```powershell
cd "C:\Users\matteo.ingrassia\Desktop\deepsight v0.7\DeepSight"
git init
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
venv/
chroma_db/
data/
models/
__pycache__/
*.pyc
```

- [ ] **Step 3: Copy model runtime** (PowerShell, from project root)

```powershell
New-Item -ItemType Directory -Force models\qwen
Copy-Item "..\model testing\Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf" models\qwen\
Copy-Item "..\model testing\benchmark\runtime\mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf" models\qwen\
Copy-Item "..\model testing\benchmark\runtime\llama-server.exe" models\qwen\
Copy-Item "..\model testing\benchmark\runtime\*.dll" models\qwen\
```

Expected: `models/qwen/` contains the two GGUFs, `llama-server.exe`, and ~20 DLLs.

- [ ] **Step 4: Edit `src/config.py`** — replace the CLIP/OCR model block (lines 22–31: `NOME_MODELLO_CLIP`, `NOME_MODELLO_CLIP_TESTO_MULTILINGUE`, `LINGUE_EASYOCR` — keep `NOME_MODELLO_WHISPER`) with:

```python
# Configurazione dei modelli di Intelligenza Artificiale
NOME_MODELLO_WHISPER = "base"  # opzioni disponibili: "tiny", "base", "small", "medium"

# --- Qwen3-VL-Embedding-2B (embedding multimodali immagine+testo, via llama-server) ---
# Sostituisce CLIP (ricerca semantica, similarità, tag) ed EasyOCR (il modello "legge"
# il testo nelle immagini a livello di embedding). Gira come sottoprocesso llama-server
# su CPU, gestito pigramente da models.GestoreQwen.
DIR_MODELLI = os.path.join(DIR_BASE, "models", "qwen")
PERCORSO_LLAMA_SERVER = os.path.join(DIR_MODELLI, "llama-server.exe")
PERCORSO_MODELLO_QWEN = os.path.join(DIR_MODELLI, "Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf")
PERCORSO_MMPROJ_QWEN = os.path.join(DIR_MODELLI, "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf")
QWEN_HOST = "127.0.0.1"
QWEN_PORT = 8091
QWEN_THREADS = 0          # 0 = lascia decidere a llama-server
DIM_EMBEDDING_QWEN = 2048  # 2B = 2048-d; NON mescolare con modelli di dimensione diversa
# Istruzione di retrieval per le QUERY testuali (i documenti/immagini NON ricevono istruzione)
ISTRUZIONE_RICERCA = "Retrieve images or text relevant to the user's query."

# Prompt negativo: punteggio finale = cos(query, img) - LAMBDA * cos(negativo, img)
LAMBDA_PROMPT_NEGATIVO = 0.5

# Tag zero-shot con Qwen: softmax(similarita' * SCALA_LOGIT_TAG). Qwen non ha la
# logit_scale appresa di CLIP: la scala e' una costante da tarare (vedi calibra_soglie.py).
SCALA_LOGIT_TAG = 40.0
```

On Linux the exe name differs; make `PERCORSO_LLAMA_SERVER` OS-aware:

```python
_NOME_SERVER = "llama-server.exe" if os.name == "nt" else "llama-server"
PERCORSO_LLAMA_SERVER = os.path.join(DIR_MODELLI, _NOME_SERVER)
```

- [ ] **Step 5: Update the similarity-threshold comment block** in `config.py` (lines 53–69). Keep `FATTORE_SIGMA_RICERCA_TESTO = 2.5` (mechanism survives; constant to re-measure) and change `SOGLIA_SIMILARITA_IMMAGINE`:

```python
# ATTENZIONE (v0.7): le soglie sotto erano tarate sui coseni di CLIP. Qwen3-VL vive su
# una scala diversa: vanno RIMISURATE su dati reali con src/calibra_soglie.py.
FATTORE_SIGMA_RICERCA_TESTO = 2.5
# Immagine->immagine con Qwen: valore iniziale prudente, da ricalibrare.
SOGLIA_SIMILARITA_IMMAGINE = 0.50
```

- [ ] **Step 6: Verify config loads**

Run (from `src/`): `..\venv\Scripts\python.exe -c "import config; print(config.PERCORSO_MODELLO_QWEN)"`
Expected: prints the GGUF path, no exceptions.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: git init, cartella modelli qwen, config v0.7"
```

---

### Task 2: `src/qwen_client.py` — vendored llama-server client + preprocessing

**Files:**
- Create: `src/qwen_client.py`
- Test: `src/test_qwen_client.py`

**Interfaces:**
- Produces: `qwen_client.smart_resize(height, width, factor=32, min_pixels=4096, max_pixels=1843200) -> (h, w)`; `qwen_client.pil_to_base64_qwen(img: PIL.Image) -> str`; `class qwen_client.LlamaServer` with `.avvia()` (start subprocess + wait healthy), `.ferma()`, `.embed_text(text, instruction=None) -> np.ndarray(2048,)`, `.embed_image_b64(b64) -> np.ndarray(2048,)`, `.base_url`; `qwen_client.EmbedError(RuntimeError)`.

- [ ] **Step 1: Write the failing test** — `src/test_qwen_client.py` (pure parts only; the live-server check lives in `test_models.py`, Task 9):

```python
"""Verifica smart_resize e la conversione base64 (senza server)."""
from PIL import Image
import qwen_client

def esegui_test():
    # multipli di 32, aspect ratio preservata
    h, w = qwen_client.smart_resize(1080, 1920)
    assert h % 32 == 0 and w % 32 == 0, (h, w)
    assert h * w <= 1843200
    # immagine minuscola: portata almeno a min_pixels
    h2, w2 = qwen_client.smart_resize(10, 10)
    assert h2 % 32 == 0 and w2 % 32 == 0 and h2 * w2 >= 4096
    # base64 di una PIL RGB
    img = Image.new("RGB", (100, 60), "red")
    b64 = qwen_client.pil_to_base64_qwen(img)
    assert isinstance(b64, str) and len(b64) > 100
    print("test_qwen_client: OK")

if __name__ == "__main__":
    esegui_test()
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `src/`): `..\venv\Scripts\python.exe test_qwen_client.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'qwen_client'`

- [ ] **Step 3: Create `src/qwen_client.py`** — merge of the two tested benchmark files, verbatim logic. Copy from `model testing/benchmark/preprocess.py`: `smart_resize` unchanged. Copy from `model testing/benchmark/embed.py`: `MEDIA_MARKER`, `EmbedError`, `LlamaServer` with these adaptations (everything else byte-identical):

1. Module docstring: keep the "Confirmed working recipe" block.
2. Replace the context-manager lifecycle with explicit methods (the app keeps the server alive for the whole session):
   - `__enter__` body → method `avvia(self)` (same body, returns `self`).
   - `__exit__` body → method `ferma(self)` (same body, signature `def ferma(self):`).
3. Log location: replace the `log_dir = Path(self.model_path).resolve().parent / "benchmark" / "out"` line in `avvia` with:

```python
        import config
        log_dir = Path(config.DIR_DATI) / "log"
```

4. Drop `selftest` and add the PIL-based preprocessing helpers (the pipeline works with frame files that may need EXIF orientation):

```python
def pil_to_base64_qwen(img, factor=32, min_pixels=4096, max_pixels=1843200):
    """PIL -> RGB -> smart_resize (lati multipli di 32) -> base64 PNG per /embeddings."""
    from PIL import Image, ImageOps
    import base64, io
    img = ImageOps.exif_transpose(img).convert("RGB")
    new_h, new_w = smart_resize(img.height, img.width, factor, min_pixels, max_pixels)
    if (new_w, new_h) != (img.width, img.height):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
```

Keep `embed_text`, `embed_image_b64`, `_post_embeddings`, `_extract_vector`, `wait_healthy`, `_cmd` exactly as in the benchmark version.

- [ ] **Step 4: Run test to verify it passes**

Run: `..\venv\Scripts\python.exe test_qwen_client.py`
Expected: `test_qwen_client: OK`

- [ ] **Step 5: Commit**

```bash
git add src/qwen_client.py src/test_qwen_client.py
git commit -m "feat: client llama-server Qwen3-VL vendored dal benchmark testato"
```

---

### Task 3: `models.py` — `GestoreQwen` lazy singleton (additive)

**Files:**
- Modify: `src/models.py` (add class + accessor; do NOT remove CLIP/OCR yet — callers migrate in later tasks)

**Interfaces:**
- Consumes: `qwen_client.LlamaServer`, `qwen_client.pil_to_base64_qwen`, config Task 1 keys.
- Produces: `gestore.ottieni_qwen() -> GestoreQwen` with `.ottieni_embedding_testo(testo, con_istruzione=True) -> np.ndarray(2048,)` (L2-normalized), `.ottieni_embedding_immagine(sorgente) -> np.ndarray(2048,)` where `sorgente` is a PIL image **or** a file path; raises `qwen_client.EmbedError` if the server is unreachable. `GestoreModelli.libera_memoria()` also stops the server.

- [ ] **Step 1: Add `GestoreQwen` to `src/models.py`** (after `GestoreCLIP`):

```python
class GestoreQwen:
    """Embedding multimodali Qwen3-VL-2B tramite sottoprocesso llama-server (CPU).

    Il server viene avviato pigramente al primo embedding e fermato da
    libera_memoria() o alla chiusura del processo (atexit).
    """

    def __init__(self):
        import qwen_client
        self._qc = qwen_client
        self.server = qwen_client.LlamaServer(
            exe=config.PERCORSO_LLAMA_SERVER,
            model_path=config.PERCORSO_MODELLO_QWEN,
            mmproj_path=config.PERCORSO_MMPROJ_QWEN,
            host=config.QWEN_HOST,
            port=config.QWEN_PORT,
            threads=config.QWEN_THREADS,
        )
        for percorso in (config.PERCORSO_LLAMA_SERVER, config.PERCORSO_MODELLO_QWEN,
                         config.PERCORSO_MMPROJ_QWEN):
            if not os.path.exists(percorso):
                raise FileNotFoundError(
                    f"File del modello Qwen mancante: {percorso}. "
                    "Copia i file in models/qwen/ (vedi README).")
        print("Avvio llama-server (Qwen3-VL-Embedding-2B)...")
        self.server.avvia()
        import atexit
        atexit.register(self.server.ferma)
        print("llama-server pronto.")

    def ottieni_embedding_testo(self, testo, con_istruzione=True):
        """Embedding 2048-d normalizzato di un testo. Le QUERY di ricerca usano
        l'istruzione di retrieval; i testi 'documento' (es. nomi di categorie) no."""
        istruzione = config.ISTRUZIONE_RICERCA if con_istruzione else None
        return self.server.embed_text(testo, instruction=istruzione)

    def ottieni_embedding_immagine(self, sorgente):
        """Embedding 2048-d normalizzato di un'immagine (percorso file o PIL)."""
        from PIL import Image
        img = Image.open(sorgente) if isinstance(sorgente, str) else sorgente
        return self.server.embed_image_b64(self._qc.pil_to_base64_qwen(img))
```

- [ ] **Step 2: Wire into `GestoreModelli`** — in `__init__` add `self._qwen = None`; add accessor next to the others; extend `libera_memoria`:

```python
    def ottieni_qwen(self) -> GestoreQwen:
        with self._lock:
            if self._qwen is None:
                self._qwen = GestoreQwen()
        return self._qwen

    # dentro libera_memoria(), prima di gc.collect():
        if self._qwen is not None:
            self._qwen.server.ferma()
        self._qwen = None
```

- [ ] **Step 3: Verify (live smoke, requires model files from Task 1)**

Run (from `src/`):
```
..\venv\Scripts\python.exe -c "from models import gestore; import numpy as np; q=gestore.ottieni_qwen(); v=q.ottieni_embedding_testo('un gatto'); print(v.shape, round(float(np.linalg.norm(v)),3))"
```
Expected: `(2048,) 1.0` (server startup takes some seconds first).

- [ ] **Step 4: Commit**

```bash
git add src/models.py
git commit -m "feat: GestoreQwen, singleton pigro per llama-server"
```

---

### Task 4: `database.py` — migration, stage columns, persons, settings, queue queries

**Files:**
- Modify: `src/database.py`
- Test: `src/test_database_v07.py`

**Interfaces:**
- Produces (consumed by Tasks 5–8):
  - `leggi_impostazione(chiave, default=None) -> str|None`, `scrivi_impostazione(chiave, valore)`
  - `crea_frame(id_media, indice_frame, secondi_timestamp, percorso_immagine) -> id_frame` (row with NULL embedding, empty tags)
  - `aggiorna_embedding_frame(id_frame, id_media, embedding, oggetti)` (BLOB + tags + Chroma upsert)
  - `ottieni_frame_di_media(id_media) -> list[dict]` with keys `id, image_path, clip_embedding_presente(bool)`
  - `imposta_stato_stadio(id_media, colonna, valore)` where colonna ∈ `{"stato_embedding","stato_volti","stato_trascrizione"}`
  - `prossimo_media_in_coda() -> dict|None` (full media_items row-dict; preparation first, then stages)
  - `conteggio_coda() -> dict` keys: `da_preparare, embedding, volti, trascrizione, falliti`
  - `elimina_volti_di_media(id_media)` (rows + crop files + Chroma vectors — for idempotent stage rerun)
  - Persons: `crea_persona() -> id`, `assegna_volto_a_persona(id_volto, id_persona)`, `ottieni_persone() -> list[dict]` (`id, name, n_volti, n_media, crop_path` of first face; prunes empty persons), `ottieni_embedding_volti_per_persona() -> dict[id_persona, list[np.ndarray]]`, `ottieni_media_di_persona(id_persona) -> list[dict]` (media rows), `rinomina_persona(id_persona, nome)`, `unisci_persone(id_da, id_a)`, `azzera_persone()` (for re-cluster).
- The Chroma frames collection is renamed `clip_frames` → `qwen_frames` (2048-d vectors; the old 512-d collection dir is abandoned on disk, harmless).

- [ ] **Step 1: Write the failing test** — `src/test_database_v07.py`:

```python
"""Migrazione v0.7: colonne di stadio, persone, impostazioni, coda."""
import os, tempfile, numpy as np

# DB temporaneo PRIMA di importare i moduli (config crea le cartelle al primo import)
os.environ.setdefault("DEEPSIGHT_TEST", "1")
import config
config.PERCORSO_DB = os.path.join(tempfile.mkdtemp(), "test.db")
import database
# Il percorso di ChromaDB NON e' configurabile: si azzerano gli handle per non
# scrivere vettori di prova nell'indice reale (tutti i chiamanti hanno il fallback).
database._store_frame = None
database._store_volti = None

def esegui_test():
    database.inizializza_db()
    # impostazioni
    assert database.leggi_impostazione("coda_in_pausa", "0") == "0"
    database.scrivi_impostazione("coda_in_pausa", "1")
    assert database.leggi_impostazione("coda_in_pausa") == "1"

    # media + frame senza embedding, poi aggiornato
    id_media = database.aggiungi_elemento_multimediale("/tmp/x.jpg", "x.jpg", "image", 1, "hash1")
    id_frame = database.crea_frame(id_media, 0, 0.0, "/tmp/x.jpg")
    frames = database.ottieni_frame_di_media(id_media)
    assert len(frames) == 1 and frames[0]["clip_embedding_presente"] is False
    emb = np.ones(config.DIM_EMBEDDING_QWEN, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    database.aggiorna_embedding_frame(id_frame, id_media, emb, ["cat"])
    assert database.ottieni_frame_di_media(id_media)[0]["clip_embedding_presente"] is True

    # coda: media appena registrato (processed=0) -> preparazione
    lavoro = database.prossimo_media_in_coda()
    assert lavoro is not None and lavoro["id"] == id_media and lavoro["processed"] == 0
    database.aggiorna_stato_elaborazione(id_media=id_media, stato_elaborazione=1)
    lavoro = database.prossimo_media_in_coda()  # ora tocca agli stadi AI
    assert lavoro["id"] == id_media
    for col in ("stato_embedding", "stato_volti", "stato_trascrizione"):
        database.imposta_stato_stadio(id_media, col, 1)
    assert database.prossimo_media_in_coda() is None
    conteggi = database.conteggio_coda()
    assert conteggi["da_preparare"] == 0 and conteggi["embedding"] == 0

    # persone
    id_p = database.crea_persona()
    id_volto = database.aggiungi_volto(id_media, id_frame, "/tmp/f.jpg",
                                       np.ones(512, dtype=np.float32) / np.sqrt(512),
                                       [0, 0, 10, 10])
    database.assegna_volto_a_persona(id_volto, id_p)
    per_persona = database.ottieni_embedding_volti_per_persona()
    assert id_p in per_persona and len(per_persona[id_p]) == 1
    database.rinomina_persona(id_p, "Mario")
    persone = database.ottieni_persone()
    assert persone[0]["name"] == "Mario" and persone[0]["n_volti"] == 1
    media_p = database.ottieni_media_di_persona(id_p)
    assert len(media_p) == 1 and media_p[0]["id"] == id_media
    print("test_database_v07: OK")

if __name__ == "__main__":
    esegui_test()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\venv\Scripts\python.exe test_database_v07.py`
Expected: FAIL with `AttributeError: module 'database' has no attribute 'leggi_impostazione'`

- [ ] **Step 3: Implement in `src/database.py`**

3a. Rename the frames collection (line 13): `ottieni_archivio_vettoriale("clip_frames")` → `ottieni_archivio_vettoriale("qwen_frames")`.

3b. In `inizializza_db()` add tables + migration before `_sincronizza_indici_vettoriali()`:

```python
    # Tabella chiave/valore per lo stato runtime (es. pausa della coda)
    cursore.execute("""
    CREATE TABLE IF NOT EXISTS impostazioni (
        chiave TEXT PRIMARY KEY,
        valore TEXT
    );
    """)

    # Tabella delle persone (raggruppamento dei volti per il Face Database)
    cursore.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    );
    """)

    connessione.commit()
    _migra_schema_v07(connessione)
    connessione.close()
    _sincronizza_indici_vettoriali()
```

3c. Add the migration function:

```python
def _migra_schema_v07(connessione):
    """Migrazione v0.7: colonne di stadio su media_items, person_id sui volti,
    azzeramento degli embedding CLIP 512-d (incompatibili con Qwen 2048-d).

    Eseguita UNA volta sola (flag in impostazioni): il backfill degli stadi
    marcherebbe come 'fatti' gli stadi legittimamente pendenti dei media v0.7."""
    cursore = connessione.cursor()
    cursore.execute("SELECT valore FROM impostazioni WHERE chiave = 'migrazione_v07_fatta'")
    if cursore.fetchone():
        return

    colonne_media = {r[1] for r in cursore.execute("PRAGMA table_info(media_items)")}
    for colonna in ("stato_embedding", "stato_volti", "stato_trascrizione"):
        if colonna not in colonne_media:
            cursore.execute(f"ALTER TABLE media_items ADD COLUMN {colonna} INTEGER DEFAULT 0")

    colonne_volti = {r[1] for r in cursore.execute("PRAGMA table_info(faces)")}
    if "person_id" not in colonne_volti:
        cursore.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER REFERENCES persons(id)")
        cursore.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")

    # Embedding della vecchia era CLIP (512 float32 = 2048 byte): vanno rigenerati.
    # Si azzerano i BLOB e si rimettono in coda i media interessati.
    cursore.execute("SELECT COUNT(*) FROM media_frames WHERE clip_embedding IS NOT NULL "
                    "AND LENGTH(clip_embedding) != ?", (config.DIM_EMBEDDING_QWEN * 4,))
    if cursore.fetchone()[0] > 0:
        print("Migrazione: trovati embedding CLIP 512-d, verranno rigenerati con Qwen.")
        cursore.execute("""
            UPDATE media_items SET stato_embedding = 0 WHERE id IN (
                SELECT DISTINCT media_id FROM media_frames
                WHERE clip_embedding IS NOT NULL AND LENGTH(clip_embedding) != ?)
        """, (config.DIM_EMBEDDING_QWEN * 4,))
        cursore.execute("UPDATE media_frames SET clip_embedding = NULL "
                        "WHERE clip_embedding IS NOT NULL AND LENGTH(clip_embedding) != ?",
                        (config.DIM_EMBEDDING_QWEN * 4,))
    # Archivi pre-v0.7 gia' elaborati: volti e trascrizione esistono gia'
    cursore.execute("UPDATE media_items SET stato_volti = 1, stato_trascrizione = 1 "
                    "WHERE processed = 1 AND stato_volti = 0 AND id IN "
                    "(SELECT DISTINCT media_id FROM media_frames)")
    cursore.execute("INSERT INTO impostazioni (chiave, valore) VALUES ('migrazione_v07_fatta', '1')")
    connessione.commit()
```

3d. Add settings, frame, stage, queue helpers:

```python
def leggi_impostazione(chiave, default=None):
    connessione = ottieni_connessione()
    riga = connessione.execute("SELECT valore FROM impostazioni WHERE chiave = ?", (chiave,)).fetchone()
    connessione.close()
    return riga[0] if riga else default

def scrivi_impostazione(chiave, valore):
    connessione = ottieni_connessione()
    connessione.execute("INSERT INTO impostazioni (chiave, valore) VALUES (?, ?) "
                        "ON CONFLICT(chiave) DO UPDATE SET valore = excluded.valore",
                        (chiave, str(valore)))
    connessione.commit()
    connessione.close()

def crea_frame(id_media, indice_frame, secondi_timestamp, percorso_immagine):
    """Registra un frame SENZA embedding (lo stadio embedding lo riempira' poi)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("""
    INSERT INTO media_frames (media_id, frame_index, timestamp_seconds, image_path, ocr_text, objects, clip_embedding)
    VALUES (?, ?, ?, ?, '', '[]', NULL)
    """, (id_media, indice_frame, secondi_timestamp, percorso_immagine))
    id_frame = cursore.lastrowid
    connessione.commit()
    connessione.close()
    return id_frame

def aggiorna_embedding_frame(id_frame, id_media, embedding, oggetti):
    """Scrive embedding Qwen (colonna storica 'clip_embedding') e tag, e aggiorna Chroma."""
    connessione = ottieni_connessione()
    connessione.execute("UPDATE media_frames SET clip_embedding = ?, objects = ? WHERE id = ?",
                        (serializza_vettore(embedding), json.dumps(list(oggetti)), id_frame))
    connessione.commit()
    connessione.close()
    if _store_frame is not None:
        try:
            _store_frame.aggiungi_o_aggiorna([id_frame], [embedding], [{"media_id": id_media}])
        except Exception as errore:
            print(f"Errore inserimento vettoriale frame {id_frame}: {errore}")

def ottieni_frame_di_media(id_media):
    connessione = ottieni_connessione()
    righe = connessione.execute(
        "SELECT id, image_path, clip_embedding IS NOT NULL FROM media_frames "
        "WHERE media_id = ? ORDER BY frame_index", (id_media,)).fetchall()
    connessione.close()
    return [{"id": r[0], "image_path": r[1], "clip_embedding_presente": bool(r[2])} for r in righe]

def imposta_stato_stadio(id_media, colonna, valore):
    assert colonna in ("stato_embedding", "stato_volti", "stato_trascrizione"), colonna
    connessione = ottieni_connessione()
    connessione.execute(f"UPDATE media_items SET {colonna} = ? WHERE id = ?", (valore, id_media))
    connessione.commit()
    connessione.close()

def prossimo_media_in_coda():
    """Prossimo elemento da lavorare: prima le preparazioni (processed=0, veloci:
    miniatura+EXIF+frame), poi gli stadi AI pendenti. None se la coda e' vuota."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("SELECT * FROM media_items WHERE processed = 0 ORDER BY id LIMIT 1")
    riga = cursore.fetchone()
    if riga is None:
        cursore.execute("""SELECT * FROM media_items WHERE processed = 1
            AND (stato_embedding = 0 OR stato_volti = 0 OR stato_trascrizione = 0)
            ORDER BY id LIMIT 1""")
        riga = cursore.fetchone()
    colonne = [c[0] for c in cursore.description] if riga else []
    connessione.close()
    return dict(zip(colonne, riga)) if riga else None

def conteggio_coda():
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    def _conta(sql):
        cursore.execute(sql)
        return cursore.fetchone()[0]
    conteggi = {
        "da_preparare": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 0"),
        "embedding": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 1 AND stato_embedding = 0"),
        "volti": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 1 AND stato_volti = 0"),
        "trascrizione": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 1 AND stato_trascrizione = 0"),
        "falliti": _conta("SELECT COUNT(*) FROM media_items WHERE processed = -1 "
                          "OR stato_embedding = -1 OR stato_volti = -1 OR stato_trascrizione = -1"),
    }
    connessione.close()
    return conteggi

def elimina_volti_di_media(id_media):
    """Rimuove volti (righe, ritagli su disco, vettori Chroma) di un media: rende
    idempotente la riesecuzione dello stadio volti dopo un fallimento."""
    connessione = ottieni_connessione()
    righe = connessione.execute("SELECT id, crop_path FROM faces WHERE media_id = ?", (id_media,)).fetchall()
    connessione.execute("DELETE FROM faces WHERE media_id = ?", (id_media,))
    connessione.commit()
    connessione.close()
    if _store_volti is not None and righe:
        try:
            _store_volti.elimina([r[0] for r in righe])
        except Exception as errore:
            print(f"Errore rimozione vettori volto per media {id_media}: {errore}")
    for _, percorso in righe:
        if percorso and os.path.exists(percorso):
            try:
                os.remove(percorso)
            except OSError:
                pass
```

3e. Add persons helpers:

```python
def crea_persona():
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("INSERT INTO persons (name) VALUES (NULL)")
    id_persona = cursore.lastrowid
    connessione.commit()
    connessione.close()
    return id_persona

def assegna_volto_a_persona(id_volto, id_persona):
    connessione = ottieni_connessione()
    connessione.execute("UPDATE faces SET person_id = ? WHERE id = ?", (id_persona, id_volto))
    connessione.commit()
    connessione.close()

def ottieni_embedding_volti_per_persona():
    """{id_persona: [embedding, ...]} per il calcolo dei centroidi (persone.py)."""
    connessione = ottieni_connessione()
    righe = connessione.execute(
        "SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL").fetchall()
    connessione.close()
    per_persona = {}
    for id_persona, blob in righe:
        emb = deserializza_vettore(blob)
        if emb is not None:
            per_persona.setdefault(id_persona, []).append(emb)
    return per_persona

def ottieni_persone():
    """Elenco persone con conteggi e ritaglio rappresentativo. Elimina le persone
    rimaste senza volti (es. dopo la cancellazione dei media)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("DELETE FROM persons WHERE id NOT IN "
                    "(SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)")
    connessione.commit()
    cursore.execute("""
        SELECT p.id, p.name, COUNT(f.id), COUNT(DISTINCT f.media_id), MIN(f.crop_path)
        FROM persons p JOIN faces f ON f.person_id = p.id
        GROUP BY p.id, p.name ORDER BY COUNT(f.id) DESC
    """)
    righe = cursore.fetchall()
    connessione.close()
    return [{"id": r[0], "name": r[1], "n_volti": r[2], "n_media": r[3], "crop_path": r[4]}
            for r in righe]

def ottieni_media_di_persona(id_persona):
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("""
        SELECT DISTINCT m.* FROM media_items m JOIN faces f ON f.media_id = m.id
        WHERE f.person_id = ? ORDER BY m.creation_date DESC
    """, (id_persona,))
    righe = cursore.fetchall()
    colonne = [c[0] for c in cursore.description]
    connessione.close()
    return [dict(zip(colonne, r)) for r in righe]

def rinomina_persona(id_persona, nome):
    connessione = ottieni_connessione()
    connessione.execute("UPDATE persons SET name = ? WHERE id = ?", (nome or None, id_persona))
    connessione.commit()
    connessione.close()

def unisci_persone(id_da, id_a):
    """Sposta tutti i volti di id_da su id_a (il nome di id_a prevale) ed elimina id_da."""
    connessione = ottieni_connessione()
    connessione.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (id_a, id_da))
    connessione.execute("DELETE FROM persons WHERE id = ?", (id_da,))
    connessione.commit()
    connessione.close()

def azzera_persone():
    """Scollega tutti i volti e svuota persons (per il re-clustering completo)."""
    connessione = ottieni_connessione()
    connessione.execute("UPDATE faces SET person_id = NULL")
    connessione.execute("DELETE FROM persons")
    connessione.commit()
    connessione.close()
```

3f. `aggiungi_frame_multimediale` stays (used by `_sincronizza_indici_vettoriali` path indirectly? no — it is only called from processor; it will be removed in Task 6 when the processor stops calling it. Leave it for now so the app still runs).

- [ ] **Step 4: Run tests**

Run: `..\venv\Scripts\python.exe test_database_v07.py` → `test_database_v07: OK`
Also run the existing check: `..\venv\Scripts\python.exe test_ricerca_vettoriale.py` → still passes (dimension-agnostic).

- [ ] **Step 5: Commit**

```bash
git add src/database.py src/test_database_v07.py
git commit -m "feat: schema v0.7 (stadi, persone, impostazioni, coda) + migrazione"
```

---

### Task 5: `src/persone.py` — clustering dei volti

**Files:**
- Create: `src/persone.py`
- Test: `src/test_persone.py`

**Interfaces:**
- Consumes: `database.ottieni_embedding_volti_per_persona`, `database.crea_persona`, `database.assegna_volto_a_persona`, `database.azzera_persone`, `database.carica_tutti_embedding_volti`, `config.SOGLIA_SIMILARITA_VOLTI`.
- Produces: `persone.assegna_persona(embedding, id_volto) -> id_persona` (greedy: joins nearest centroid ≥ threshold else creates person); `persone.ricalcola_tutti_cluster() -> int` (DBSCAN over all faces, preserves names by majority, returns #persons).

- [ ] **Step 1: Write the failing test** — `src/test_persone.py`:

```python
"""Clustering dei volti: assegnazione greedy e re-cluster DBSCAN."""
import os, tempfile
import numpy as np

import config
config.PERCORSO_DB = os.path.join(tempfile.mkdtemp(), "test.db")
import database
database._store_frame = None  # non toccare l'indice Chroma reale (vedi test_database_v07)
database._store_volti = None
import persone

def _vettore(seme):
    rng = np.random.default_rng(seme)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)

def _variante(base, rumore=0.05, seme=0):
    rng = np.random.default_rng(seme)
    v = base + rumore * rng.normal(size=512).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)

def esegui_test():
    database.inizializza_db()
    id_media = database.aggiungi_elemento_multimediale("/tmp/y.jpg", "y.jpg", "image", 1, "hash_p")
    # carica_tutti_embedding_volti filtra su processed=1: il media va marcato elaborato
    database.aggiorna_stato_elaborazione(id_media=id_media, stato_elaborazione=1)
    id_frame = database.crea_frame(id_media, 0, 0.0, "/tmp/y.jpg")

    alice = _vettore(1)
    bruno = _vettore(2)

    # due volti quasi identici -> stessa persona; uno diverso -> nuova persona
    v1 = database.aggiungi_volto(id_media, id_frame, "/tmp/a1.jpg", alice, [0,0,1,1])
    p1 = persone.assegna_persona(alice, v1)
    v2 = database.aggiungi_volto(id_media, id_frame, "/tmp/a2.jpg", _variante(alice, seme=3), [0,0,1,1])
    p2 = persone.assegna_persona(_variante(alice, seme=3), v2)
    v3 = database.aggiungi_volto(id_media, id_frame, "/tmp/b1.jpg", bruno, [0,0,1,1])
    p3 = persone.assegna_persona(bruno, v3)
    assert p1 == p2, "varianti della stessa faccia devono unirsi"
    assert p3 != p1, "facce diverse devono separarsi"

    # re-cluster completo: stesso risultato (2 persone), nome preservato
    database.rinomina_persona(p1, "Alice")
    n = persone.ricalcola_tutti_cluster()
    assert n == 2, n
    nomi = {p["name"] for p in database.ottieni_persone()}
    assert "Alice" in nomi
    print("test_persone: OK")

if __name__ == "__main__":
    esegui_test()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\venv\Scripts\python.exe test_persone.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'persone'`

- [ ] **Step 3: Create `src/persone.py`**

```python
"""Raggruppamento dei volti in persone (Face Database).

Strategia incrementale: un volto nuovo entra nella persona con il centroide piu'
simile se il coseno supera config.SOGLIA_SIMILARITA_VOLTI, altrimenti fonda una
persona nuova. Il re-cluster completo (DBSCAN) corregge la deriva del greedy.
"""
# ponytail: assegnazione greedy sul centroide; se le fusioni sbagliate diventano
# un problema, passare a re-cluster automatico periodico (HAC/DBSCAN).
import numpy as np

import config
import database


def _centroidi():
    """{id_persona: centroide normalizzato} dai volti gia' assegnati."""
    risultato = {}
    for id_persona, embedding in database.ottieni_embedding_volti_per_persona().items():
        c = np.mean(np.stack(embedding), axis=0)
        norma = np.linalg.norm(c)
        if norma > 0:
            risultato[id_persona] = c / norma
    return risultato


def assegna_persona(embedding, id_volto):
    """Assegna il volto alla persona piu' vicina (o ne crea una). Ritorna id_persona."""
    migliore_id, migliore_sim = None, -1.0
    for id_persona, centroide in _centroidi().items():
        sim = float(np.dot(embedding, centroide))
        if sim > migliore_sim:
            migliore_id, migliore_sim = id_persona, sim
    if migliore_id is None or migliore_sim < config.SOGLIA_SIMILARITA_VOLTI:
        migliore_id = database.crea_persona()
    database.assegna_volto_a_persona(id_volto, migliore_id)
    return migliore_id


def ricalcola_tutti_cluster():
    """Re-clustering completo con DBSCAN (metrica coseno). Preserva i nomi
    esistenti assegnandoli al cluster che contiene la maggioranza dei volti
    della vecchia persona. Ritorna il numero di persone risultanti."""
    from sklearn.cluster import DBSCAN

    volti = database.carica_tutti_embedding_volti()
    if not volti:
        database.azzera_persone()
        return 0

    # nome della vecchia persona per ogni volto (per preservarlo dopo)
    vecchi_nomi = {p["id"]: p["name"] for p in database.ottieni_persone() if p["name"]}
    connessione = database.ottieni_connessione()
    volto_a_vecchia_persona = dict(connessione.execute(
        "SELECT id, person_id FROM faces WHERE person_id IS NOT NULL").fetchall())
    connessione.close()

    matrice = np.stack([v["embedding"] for v in volti])
    # eps in distanza coseno = 1 - soglia di similarita'
    etichette = DBSCAN(eps=1.0 - config.SOGLIA_SIMILARITA_VOLTI, min_samples=1,
                       metric="cosine").fit_predict(matrice)

    database.azzera_persone()
    persona_per_etichetta = {}
    voti_nome = {}  # etichetta -> {nome: conteggio}
    for volto, etichetta in zip(volti, etichette):
        if etichetta not in persona_per_etichetta:
            persona_per_etichetta[etichetta] = database.crea_persona()
        database.assegna_volto_a_persona(volto["face_id"], persona_per_etichetta[etichetta])
        vecchio_nome = vecchi_nomi.get(volto_a_vecchia_persona.get(volto["face_id"]))
        if vecchio_nome:
            conteggi = voti_nome.setdefault(etichetta, {})
            conteggi[vecchio_nome] = conteggi.get(vecchio_nome, 0) + 1

    for etichetta, conteggi in voti_nome.items():
        nome_maggioranza = max(conteggi, key=conteggi.get)
        database.rinomina_persona(persona_per_etichetta[etichetta], nome_maggioranza)

    return len(persona_per_etichetta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `..\venv\Scripts\python.exe test_persone.py` → `test_persone: OK`

- [ ] **Step 5: Commit**

```bash
git add src/persone.py src/test_persone.py
git commit -m "feat: clustering volti in persone (greedy + re-cluster DBSCAN)"
```

---

### Task 6: `processor.py` — staged pipeline + pausable worker

**Files:**
- Modify: `src/processor.py`

**Interfaces:**
- Consumes: `gestore.ottieni_qwen()`, `gestore.ottieni_volti()`, `gestore.ottieni_whisper()`, `persone.assegna_persona`, database helpers from Task 4.
- Produces (consumed by app.py in Task 7):
  - `avvia_lavoratore()` — idempotent, starts the daemon worker thread and wakes it
  - `stato_coda() -> dict` — `conteggio_coda()` plus `{"in_corso": str|None, "in_pausa": bool, "eta_secondi": float|None}`
  - `metti_in_pausa()` / `riprendi()` — flip the `coda_in_pausa` setting (+ wake)
  - `classifica_tag(embedding_immagine) -> list[str]` (new signature — no CLIP handle)
  - `registra_file` unchanged; `aggiungi_e_elabora_file(percorso, hash_precalcolato=None)` now = register only + wake worker (kept for `scansiona_cartella_condivisa` / `importa_file_intrusi` callers).
- Removes: `elabora_immagine`, `elabora_video`, `elabora_file_registrato`, `avvia_elaborazione_in_background`, `stato_elaborazione_background`, `riprendi_elaborazioni_interrotte`, `_ciclo_elaborazione_background`, `_coda_elaborazione` and related globals, all OCR usage, `ottieni_embedding_categorie`'s CLIP dependency.

- [ ] **Step 1: Replace zero-shot tagging** (delete `ottieni_embedding_categorie` and old `classifica_tag`, keep `CATEGORIE`):

```python
_embedding_categorie_precomputati = None

def _embedding_categorie(qwen):
    """Embedding testuali delle categorie (senza istruzione: sono 'documenti')."""
    global _embedding_categorie_precomputati
    if _embedding_categorie_precomputati is None:
        _embedding_categorie_precomputati = [
            (cat, qwen.ottieni_embedding_testo(f"a photo of a {cat}", con_istruzione=False))
            for cat in CATEGORIE
        ]
    return _embedding_categorie_precomputati

def classifica_tag(embedding_immagine, soglia=None, max_tag=None):
    """Tag zero-shot: softmax(coseni * config.SCALA_LOGIT_TAG) sulle categorie comuni."""
    if soglia is None:
        soglia = config.SOGLIA_PROBABILITA_TAG
    if max_tag is None:
        max_tag = config.MASSIMO_TAG_PER_IMMAGINE
    embedding_cat = _embedding_categorie(gestore.ottieni_qwen())
    similarita = np.array([float(np.dot(embedding_immagine, e)) for _, e in embedding_cat])
    logits = similarita * config.SCALA_LOGIT_TAG
    exp_logits = np.exp(logits - np.max(logits))
    probabilita = exp_logits / np.sum(exp_logits)
    coppie = [(cat, p) for (cat, _), p in zip(embedding_cat, probabilita) if p >= soglia]
    coppie.sort(key=lambda x: x[1], reverse=True)
    return [cat for cat, _ in coppie[:max_tag]]
```

- [ ] **Step 2: Add the preparation stage** (fast, no AI — replaces the metadata half of `elabora_immagine`/`elabora_video`):

```python
def _prepara_media(elemento):
    """Stadio veloce senza AI: miniatura, EXIF/GPS, estrazione frame. Porta
    processed a 1 (visibile in galleria) o -1. Idempotente: azzera i frame parziali."""
    id_media, percorso, tipo_media = elemento["id"], elemento["file_path"], elemento["media_type"]

    # riesecuzione dopo un crash: elimina frame parziali
    connessione = database.ottieni_connessione()
    connessione.execute("DELETE FROM media_frames WHERE media_id = ?", (id_media,))
    connessione.commit()
    connessione.close()

    crea_anteprima(percorso, tipo_media)

    if tipo_media == "image":
        pil_img = ImageOps.exif_transpose(Image.open(percorso)).convert("RGB")
        larghezza, altezza = pil_img.size
        data_creazione, lat, lon = estrai_metadati_exif(percorso)
        if data_creazione is None:
            data_creazione = datetime.datetime.fromtimestamp(os.stat(percorso).st_mtime).isoformat()
        nome_localita = ottieni_nome_luogo(lat, lon)
        database.aggiorna_stato_elaborazione(
            id_media=id_media, data_creazione=data_creazione, latitudine=lat,
            longitudine=lon, nome_localita=nome_localita,
            larghezza=larghezza, altezza=altezza, stato_elaborazione=1)
        database.crea_frame(id_media, 0, 0.0, percorso)
        # le immagini non hanno audio: lo stadio trascrizione e' gia' "fatto"
        database.imposta_stato_stadio(id_media, "stato_trascrizione", 1)
    else:  # video
        proprieta = ottieni_proprieta_video(percorso)
        if not proprieta:
            raise ValueError("Impossibile leggere i metadati del video.")
        data_creazione = datetime.datetime.fromtimestamp(os.stat(percorso).st_mtime).isoformat()
        # estrazione frame a intervalli regolari (identica alla v0.6, senza AI)
        cattura = cv2.VideoCapture(percorso)
        if not cattura.isOpened():
            raise ValueError("Impossibile aprire il file video.")
        fps = proprieta["fps"]
        intervallo = int(config.INTERVALLO_FRAME_VIDEO * fps) if fps > 0 else 90
        intervallo = intervallo if intervallo > 0 else 90
        indice_frame, conteggio_estratti = 0, 0
        while cattura.isOpened():
            letto, frame = cattura.read()
            if not letto:
                break
            if indice_frame % intervallo == 0:
                timestamp = indice_frame / fps if fps > 0 else 0.0
                percorso_frame = os.path.join(config.DIR_FRAME, f"{id_media}_frame_{conteggio_estratti}.jpg")
                cv2.imwrite(percorso_frame, frame)
                database.crea_frame(id_media, conteggio_estratti, timestamp, percorso_frame)
                conteggio_estratti += 1
            indice_frame += 1
        cattura.release()
        database.aggiorna_stato_elaborazione(
            id_media=id_media, data_creazione=data_creazione,
            larghezza=proprieta["larghezza"], altezza=proprieta["altezza"],
            durata=proprieta["durata"], stato_elaborazione=1)
```

- [ ] **Step 3: Add the three AI stages**

```python
def _stadio_embedding(elemento):
    """Embedding Qwen + tag per ogni frame senza embedding (riprende dai mancanti)."""
    qwen = gestore.ottieni_qwen()
    for frame in database.ottieni_frame_di_media(elemento["id"]):
        if frame["clip_embedding_presente"]:
            continue
        emb = qwen.ottieni_embedding_immagine(frame["image_path"])
        tags = classifica_tag(emb)
        database.aggiorna_embedding_frame(frame["id"], elemento["id"], emb, tags)

def _stadio_volti(elemento):
    """MTCNN+FaceNet su ogni frame; ogni volto viene assegnato a una persona."""
    import persone
    face_rec = gestore.ottieni_volti()
    database.elimina_volti_di_media(elemento["id"])  # idempotenza su rieseguzione
    for frame in database.ottieni_frame_di_media(elemento["id"]):
        pil_img = ImageOps.exif_transpose(Image.open(frame["image_path"])).convert("RGB")
        for idx, dati_volto in enumerate(face_rec.rileva_e_codifica_volti(pil_img)):
            percorso_volto = os.path.join(
                config.DIR_VOLTI, f"{elemento['id']}_fr_{frame['id']}_face_{idx}.jpg")
            dati_volto["crop"].save(percorso_volto, "JPEG")
            id_volto = database.aggiungi_volto(
                id_media=elemento["id"], id_frame=frame["id"],
                percorso_ritaglio=percorso_volto,
                embedding=dati_volto["embedding"], riquadro=dati_volto["bbox"])
            persone.assegna_persona(dati_volto["embedding"], id_volto)

def _stadio_trascrizione(elemento):
    """Solo video: estrazione audio + Whisper."""
    whisper = gestore.ottieni_whisper()
    trascrizione = None
    percorso_audio = estrai_audio_video(elemento["file_path"])
    if percorso_audio:
        try:
            trascrizione = whisper.trascrivi(percorso_audio)
        finally:
            if os.path.exists(percorso_audio):
                os.remove(percorso_audio)
    database.aggiorna_stato_elaborazione(id_media=elemento["id"], trascrizione=trascrizione,
                                         stato_elaborazione=1)
```

- [ ] **Step 4: Replace the background-worker block** (delete everything from `_coda_elaborazione = queue.Queue()` down to `_ciclo_elaborazione_background`, and `riprendi_elaborazioni_interrotte`; remove the now-unused `import queue`):

```python
# ---------------------------------------------------------------------------
# Lavoratore in background (coda persistente su SQLite)
#
# La coda NON e' una struttura in memoria: e' la query database.prossimo_media_in_coda()
# sulle colonne di stadio. Un riavvio riprende esattamente da dove si era rimasti.
# Il thread daemon lavora un elemento alla volta e controlla la pausa tra gli stadi.
# ponytail: un solo thread lavoratore; processo separato solo se i rerun di
# Streamlit dovessero interferire.
# ---------------------------------------------------------------------------

_lavoratore = None
_lock_lavoratore = threading.Lock()
_sveglia = threading.Event()
_stato_runtime = {"in_corso": None, "durate": []}  # durate: ultimi secondi/elemento per l'ETA

def avvia_lavoratore():
    """Avvia (una sola volta per processo) il thread della coda e lo sveglia."""
    global _lavoratore
    with _lock_lavoratore:
        if _lavoratore is None or not _lavoratore.is_alive():
            _lavoratore = threading.Thread(target=_ciclo_lavoratore, daemon=True,
                                           name="deepsight-coda")
            _lavoratore.start()
    _sveglia.set()

def metti_in_pausa():
    database.scrivi_impostazione("coda_in_pausa", "1")

def riprova_falliti():
    """Rimette in coda gli elementi/stadi falliti (-1 -> 0). Ritorna quanti."""
    connessione = database.ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("UPDATE media_items SET processed = 0 WHERE processed = -1")
    n = cursore.rowcount
    for colonna in ("stato_embedding", "stato_volti", "stato_trascrizione"):
        cursore.execute(f"UPDATE media_items SET {colonna} = 0 WHERE {colonna} = -1")
        n += cursore.rowcount
    connessione.commit()
    connessione.close()
    _sveglia.set()
    return n

def riprendi():
    database.scrivi_impostazione("coda_in_pausa", "0")
    _sveglia.set()

def stato_coda():
    """Fotografia per la UI: conteggi, elemento in corso, pausa, ETA stimato."""
    conteggi = database.conteggio_coda()
    rimanenti = conteggi["da_preparare"] + conteggi["embedding"] + conteggi["volti"] + conteggi["trascrizione"]
    durate = _stato_runtime["durate"]
    eta = (sum(durate) / len(durate)) * rimanenti if durate and rimanenti else None
    return {**conteggi, "rimanenti": rimanenti, "in_corso": _stato_runtime["in_corso"],
            "in_pausa": database.leggi_impostazione("coda_in_pausa", "0") == "1",
            "eta_secondi": eta}

_STADI = (("stato_embedding", _stadio_embedding),
          ("stato_volti", _stadio_volti),
          ("stato_trascrizione", _stadio_trascrizione))

def _ciclo_lavoratore():
    import time
    while True:
        if database.leggi_impostazione("coda_in_pausa", "0") == "1":
            _sveglia.clear()
            _sveglia.wait(timeout=2.0)
            continue
        elemento = database.prossimo_media_in_coda()
        if elemento is None:
            _sveglia.clear()
            _sveglia.wait(timeout=2.0)
            continue

        _stato_runtime["in_corso"] = elemento["filename"]
        inizio = time.monotonic()
        try:
            if elemento["processed"] == 0:
                _prepara_media(elemento)
            else:
                for colonna, stadio in _STADI:
                    if elemento[colonna] != 0:
                        continue
                    if database.leggi_impostazione("coda_in_pausa", "0") == "1":
                        break  # pausa tra gli stadi: l'elemento restera' in coda
                    try:
                        stadio(elemento)
                        database.imposta_stato_stadio(elemento["id"], colonna, 1)
                    except Exception as errore:
                        print(f"Stadio {colonna} fallito per {elemento['filename']}: {errore}")
                        database.imposta_stato_stadio(elemento["id"], colonna, -1)
        except Exception as errore:
            print(f"Preparazione fallita per {elemento['filename']}: {errore}")
            database.aggiorna_stato_elaborazione(id_media=elemento["id"], stato_elaborazione=-1)
        finally:
            _stato_runtime["in_corso"] = None
            durate = _stato_runtime["durate"]
            durate.append(time.monotonic() - inizio)
            del durate[:-20]  # media mobile sugli ultimi 20 elementi
```

- [ ] **Step 5: Rewire the entry points.** Delete `elabora_file_registrato`, `elabora_immagine`, `elabora_video`. Replace `aggiungi_e_elabora_file`:

```python
def aggiungi_e_elabora_file(percorso_origine, hash_precalcolato=None):
    """Importa un file: copia in archivio + record nel DB. L'elaborazione AI
    avviene in background tramite la coda a stadi (avvia_lavoratore)."""
    id_media, percorso_archiviato, tipo_media = registra_file(percorso_origine, hash_precalcolato)
    avvia_lavoratore()
    return id_media, True
```

In `scansiona_cartella_condivisa` the call site is unchanged (it already calls `aggiungi_e_elabora_file`). In `importa_file_intrusi` likewise unchanged. Note: a re-imported duplicate returns the existing id (via the IntegrityError path in `aggiungi_elemento_multimediale`) — already handled.

> **Nota di sequenza:** questo task rimuove funzioni ancora chiamate da `app.py`
> (`riprendi_elaborazioni_interrotte`, `avvia_elaborazione_in_background`,
> `stato_elaborazione_background`): la UI Streamlit resta rotta fino al Task 7,
> che migra i chiamanti. Gli script di test non importano `app.py` e restano verdi.

- [ ] **Step 6: Quick sanity run**

Run: `..\venv\Scripts\python.exe -c "import processor; processor.avvia_lavoratore(); import time; time.sleep(1); print(processor.stato_coda())"`
Expected: prints a dict with `da_preparare: 0 ... in_pausa: False` (empty DB) and exits cleanly.

- [ ] **Step 7: Commit**

```bash
git add src/processor.py
git commit -m "feat: pipeline a stadi con lavoratore pausabile su coda SQLite"
```

---

### Task 7: `app.py` — search rewrite (Qwen + negative prompt), queue panel

**Files:**
- Modify: `src/app.py`

**Interfaces:**
- Consumes: `gestore.ottieni_qwen()`, `processor.avvia_lavoratore/metti_in_pausa/riprendi/stato_coda`, `config.LAMBDA_PROMPT_NEGATIVO`.

- [ ] **Step 1: Startup block** (app.py:36–38) — replace `processor.riprendi_elaborazioni_interrotte()` with:

```python
# Avvia il lavoratore della coda: riprende da solo gli elementi pendenti nel DB.
processor.avvia_lavoratore()
```

- [ ] **Step 2: Replace the sidebar progress fragment** (app.py:707–728, `indicatore_elaborazione_background`) with the queue panel:

```python
# Pannello coda: si auto-aggiorna ogni 2 secondi ed e' visibile da ogni pagina.
@st.fragment(run_every=2.0)
def pannello_coda():
    stato = processor.stato_coda()
    if stato["rimanenti"] == 0 and not stato["in_corso"]:
        if stato["falliti"]:
            st.warning(f"⚠️ {stato['falliti']} elementi con elaborazioni fallite")
        return
    riga = f"⚙️ **Coda elaborazione:** {stato['rimanenti']} elementi"
    if stato["in_corso"]:
        riga += f"\n\n`{stato['in_corso']}`"
    if stato["eta_secondi"]:
        riga += f"\n\n⏳ stimati {datetime.timedelta(seconds=int(stato['eta_secondi']))}"
    st.info(riga)
    st.caption(f"da preparare: {stato['da_preparare']} · embedding: {stato['embedding']} · "
               f"volti: {stato['volti']} · trascrizioni: {stato['trascrizione']}")
    if stato["in_pausa"]:
        st.warning("⏸️ Coda in pausa (l'elemento corrente viene completato)")
        if st.button("▶️ Riprendi", key="coda_riprendi", width='stretch'):
            processor.riprendi()
            st.rerun(scope="fragment")
    else:
        if st.button("⏸️ Metti in pausa", key="coda_pausa", width='stretch'):
            processor.metti_in_pausa()
            st.rerun(scope="fragment")
    if stato["falliti"]:
        if st.button(f"🔁 Riprova falliti ({stato['falliti']})", key="coda_riprova", width='stretch'):
            processor.riprova_falliti()
            st.rerun(scope="fragment")

with st.sidebar:
    pannello_coda()
```

- [ ] **Step 2b: Upload call site** (app.py:1045–1055) — the batch is already registered by `processor.registra_file` in the loop above; replace `processor.avvia_elaborazione_in_background(lotto)` with `processor.avvia_lavoratore()` and update the success message to mention the queue panel ("l'avanzamento e i controlli pausa/riprendi sono nella barra laterale"). The `lotto` list is now only used for its length in the messages.

- [ ] **Step 3: Text search tab** (app.py:1113–1139, `sotto_scheda1`) — Qwen embedding + negative prompt. Replace the body with:

```python
    with sotto_scheda1:
        st.markdown("#### Ricerca per concetto testuale")
        testo_query = st.text_input("Inserisci cosa stai cercando (es: 'una spiaggia al tramonto', 'strada con auto', 'cane che corre')", "")
        usa_negativo = st.toggle("Prompt negativo (escludi elementi)", key="tgl_negativo")
        testo_negativo = ""
        if usa_negativo:
            testo_negativo = st.text_input("Cosa NON deve comparire (es: 'persone', 'neve')", "", key="txt_negativo")

        if testo_query:
            with st.spinner("Calcolo embedding della query e ricerca vettoriale..."):
                try:
                    qwen = gestore.ottieni_qwen()
                    query_emb = qwen.ottieni_embedding_testo(testo_query)
                    soglia = database.soglia_adattiva_testo(query_emb)
                    emb_negativo = None
                    if usa_negativo and testo_negativo.strip():
                        emb_negativo = qwen.ottieni_embedding_testo(testo_negativo.strip())

                    for f, sim in database.cerca_frame_simili(query_emb):
                        if not applica_filtri(f, filtri):
                            continue
                        if sim < soglia:
                            continue
                        punteggio = sim
                        if emb_negativo is not None:
                            # penalita': gli elementi affini al prompt negativo scendono
                            punteggio = sim - config.LAMBDA_PROMPT_NEGATIVO * float(
                                np.dot(emb_negativo, f["embedding"]))
                        risultati.append((f, punteggio, "clip"))

                    risultati = deduplica_risultati(risultati)
                    risultati = risultati[:30]
                except Exception as errore:
                    st.error(f"Errore nella ricerca semantica: {errore}")
```

Add `import numpy as np` to app.py's imports (top of file).

- [ ] **Step 4: Image-similarity tab** (app.py:1157–1161) — swap the embedding call inside the button handler:

```python
            if st.button("Esegui Ricerca per Similarità"):
                with st.spinner("Elaborazione immagine di query..."):
                    qwen = gestore.ottieni_qwen()
                    st.session_state["emb_query_img"] = qwen.ottieni_embedding_immagine(immagine_pil_query)
                    st.session_state["id_file_img"] = id_file_img
```

- [ ] **Step 5: Tab 4 OCR removal** (app.py:1104–1109 tab label, 1232–1311 body). Rename the tab `"📝 OCR e Parlato"` → `"🗣️ Parlato (Whisper)"`; in the body delete the OCR query (the `righe_ocr` SELECT and its result loop) and the OCR caption text — keep only the Whisper transcription search (`righe_video` block). In the results card (app.py:1384–1385) delete the `if elemento.get("ocr_text"):` block.

- [ ] **Step 6: Search score display note.** With a negative prompt the score can go below 0; clamp for display at app.py:1344:

```python
                stringa_punteggio = "Corrispondenza Testo" if modalita.startswith("text_") else f"{max(0.0, punteggio) * 100:.1f}% Rilevanza"
```

- [ ] **Step 7: Manual verification** — start the app (`.\scripts\windows\run.bat`), upload 2–3 test images from `model testing/testing images/`, watch the queue panel count down, then run a text search with and without a negative prompt.
Expected: upload returns immediately; queue shows stages; search returns results once embeddings complete; pause button freezes the count.

- [ ] **Step 8: Commit**

```bash
git add src/app.py
git commit -m "feat: ricerca Qwen con prompt negativo, pannello coda pausabile, via OCR"
```

---

### Task 8: `app.py` — pagina "Persone" (Face Database)

**Files:**
- Modify: `src/app.py`

**Interfaces:**
- Consumes: `database.ottieni_persone`, `ottieni_media_di_persona`, `rinomina_persona`, `unisci_persone`; `persone.ricalcola_tutti_cluster`.

- [ ] **Step 1: Add the navbar button.** At app.py:757–787 change columns to `st.columns([1, 1, 1, 1, 0.35])` and add after the Ricerca button:

```python
with col_nav4:
    attivo_persone = st.session_state.selezione_menu == "👤 Persone"
    if st.button("👤 Persone", width='stretch', key="nav_persone", type="primary" if attivo_persone else "secondary"):
        st.session_state.selezione_menu = "👤 Persone"
        st.rerun()
```

(the shield popover moves to `col_nav5`).

- [ ] **Step 2: Add the page** (new `elif` before the Ricerca Avanzata block):

```python
# --- SCHEDA PERSONE (Face Database) ---
elif menu == "👤 Persone":
    st.markdown("<h1 class='main-title'>Persone</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Tutte le persone riconosciute nell'archivio. Clicca una persona per vedere i suoi contenuti e assegnarle un nome.</p>", unsafe_allow_html=True)

    import persone as modulo_persone
    lista_persone = database.ottieni_persone()

    if st.button("🔄 Ricalcola raggruppamenti", key="btn_recluster",
                 help="Riesegue il clustering di tutti i volti (i nomi vengono preservati)"):
        with st.spinner("Re-clustering dei volti in corso..."):
            n = modulo_persone.ricalcola_tutti_cluster()
        st.toast(f"Raggruppamento completato: {n} persone.", icon="✅")
        st.rerun()

    if not lista_persone:
        st.info("Nessun volto ancora rilevato: carica contenuti e attendi lo stadio 'volti' della coda.")
    else:
        id_selezionata = st.session_state.get("persona_selezionata")

        if id_selezionata is None:
            # griglia delle persone
            with st.container(key="risultati_griglia"):
                for p in lista_persone:
                    with st.container():
                        nome = p["name"] or f"Persona {p['id']}"
                        st.markdown(f"""
                        <div class="result-card"><div class="result-meta">
                            <div class="result-title">{nome}</div>
                            <div style="opacity:0.7; font-size:0.8rem;">
                                {p['n_media']} contenuti · {p['n_volti']} volti</div>
                        </div></div>""", unsafe_allow_html=True)
                        if p["crop_path"] and os.path.exists(p["crop_path"]):
                            st.image(p["crop_path"], width="stretch")
                        if st.button("Apri", key=f"apri_persona_{p['id']}", width='stretch'):
                            st.session_state["persona_selezionata"] = p["id"]
                            st.rerun()
        else:
            persona = next((p for p in lista_persone if p["id"] == id_selezionata), None)
            if persona is None:
                st.session_state.pop("persona_selezionata", None)
                st.rerun()
            if st.button("← Tutte le persone", key="btn_indietro_persone"):
                st.session_state.pop("persona_selezionata", None)
                st.rerun()

            col_p1, col_p2 = st.columns([1, 2])
            with col_p1:
                if persona["crop_path"] and os.path.exists(persona["crop_path"]):
                    st.container(key="volti_query").image(persona["crop_path"], width=160)
            with col_p2:
                nuovo_nome = st.text_input("Nome", value=persona["name"] or "", key=f"nome_p_{persona['id']}")
                if st.button("💾 Salva nome", key=f"salva_nome_{persona['id']}"):
                    database.rinomina_persona(persona["id"], nuovo_nome.strip())
                    st.toast("Nome salvato.", icon="✅")
                    st.rerun()
                altre = [p for p in lista_persone if p["id"] != persona["id"]]
                if altre:
                    etichette = {f"{p['name'] or f'Persona {p.id}'.replace('.id', str(p['id']))} (#{p['id']})": p["id"] for p in altre}
                    scelta = st.selectbox("Unisci con...", ["—"] + list(etichette), key=f"merge_sel_{persona['id']}")
                    if scelta != "—" and st.button("🔗 Unisci (i volti passano a questa persona)", key=f"merge_btn_{persona['id']}"):
                        database.unisci_persone(etichette[scelta], persona["id"])
                        st.toast("Persone unite.", icon="✅")
                        st.rerun()

            st.markdown(f"### Contenuti con {persona['name'] or f'Persona {persona['id']}'}")
            media_persona = database.ottieni_media_di_persona(persona["id"])
            with st.container(key="galleria_griglia"):
                for elemento in media_persona:
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card"><div class="result-meta">
                            <div class="result-title" title="{elemento['filename']}">{elemento['filename']}</div>
                            <div style="opacity:0.7; font-size:0.8rem;">
                                Tipo: {elemento['media_type'].upper()}<br>
                                📅 {elemento['creation_date'].split('T')[0] if elemento['creation_date'] else 'N/D'}</div>
                        </div></div>""", unsafe_allow_html=True)
                        if elemento["media_type"] == "image" and os.path.exists(elemento["file_path"]):
                            st.image(immagine_per_display(elemento["file_path"], lato_max=1000), width="stretch")
                        else:
                            anteprima = percorso_anteprima_elemento(elemento["file_path"])
                            if anteprima:
                                st.image(anteprima, width="stretch")
                        if os.path.exists(elemento["file_path"]):
                            with open(elemento["file_path"], "rb") as dati_file:
                                st.download_button("⬇️ Scarica", data=dati_file,
                                                   file_name=elemento["filename"],
                                                   mime="application/octet-stream",
                                                   key=f"persona_dl_{persona['id']}_{elemento['id']}")
```

Fix the f-string nesting in the two label spots if the Python version complains (extract `nome_persona = persona["name"] or f"Persona {persona['id']}"` to a variable first — do this, it's cleaner):

```python
            nome_persona = persona["name"] or f"Persona {persona['id']}"
```
and use `{nome_persona}` in both places; build `etichette` as:
```python
                    etichette = {f"{p['name'] or 'Persona ' + str(p['id'])} (#{p['id']})": p["id"] for p in altre}
```

- [ ] **Step 3: Manual verification** — with test images containing faces processed: Persone grid shows clusters; open one, rename, merge two, re-cluster preserves the name.

- [ ] **Step 4: Commit**

```bash
git add src/app.py
git commit -m "feat: pagina Persone (face database: nomi, unione, re-cluster)"
```

---

### Task 9: Purge CLIP/EasyOCR + update diagnostics + calibration script

**Files:**
- Modify: `src/models.py`, `src/test_models.py`, `requirements.txt`, `scripts/windows/install.ps1`, `scripts/linux/install.sh`
- Create: `src/calibra_soglie.py`

- [ ] **Step 1: Delete `GestoreCLIP` and `GestoreOCR`** from `src/models.py` (classes + `_clip`/`_ocr` fields + `ottieni_clip`/`ottieni_ocr` accessors + their lines in `libera_memoria`). Verify no references remain:

```
grep -rn "ottieni_clip\|ottieni_ocr\|GestoreCLIP\|GestoreOCR" src/
```
Expected: no matches (Tasks 6–7 already migrated the callers).

- [ ] **Step 2: Rewrite `src/test_models.py`** — same diagnostic style, three models:

```python
"""Diagnostica: verifica che i modelli AI (Qwen, FaceNet, Whisper) si carichino."""
import numpy as np
from models import gestore
import config

def esegui_test():
    print(f"Dispositivo: {gestore.dispositivo}")

    print("1/3 Qwen3-VL-Embedding (llama-server)...")
    qwen = gestore.ottieni_qwen()
    v_testo = qwen.ottieni_embedding_testo("una prova")
    assert v_testo.shape == (config.DIM_EMBEDDING_QWEN,), v_testo.shape
    assert abs(float(np.linalg.norm(v_testo)) - 1.0) < 1e-3
    print("   OK: embedding testo 2048-d normalizzato")

    print("2/3 MTCNN + FaceNet...")
    gestore.ottieni_volti()
    print("   OK")

    print("3/3 Whisper...")
    gestore.ottieni_whisper()
    print("   OK")
    print("Tutti i modelli caricati correttamente.")

if __name__ == "__main__":
    esegui_test()
```

- [ ] **Step 3: `requirements.txt`** — delete `transformers`, `sentence-transformers`, and the whole "EasyOCR Dependencies" block (`scikit-image`, `ninja`, `pyclipper`, `python-bidi`, `Shapely`); update the trailing NOTE to mention only `facenet-pytorch`. Add under Core: `scikit-learn>=1.4.0` (DBSCAN — it was previously pulled in transitively; make it explicit).

- [ ] **Step 4: Install scripts** — in `scripts/windows/install.ps1` and `scripts/linux/install.sh` remove the `easyocr --no-deps` install line (keep `facenet-pytorch --no-deps`). Add at the end of each a check that warns (does not fail) if `models/qwen/` is missing the three required files, printing where to get them.

- [ ] **Step 5: Create `src/calibra_soglie.py`** — measures the Qwen cosine distributions on the real archive so the config constants can be re-tuned:

```python
"""Misura le distribuzioni dei coseni Qwen sull'archivio reale, per tarare:
FATTORE_SIGMA_RICERCA_TESTO, SOGLIA_SIMILARITA_IMMAGINE, SCALA_LOGIT_TAG.
Uso: python calibra_soglie.py "query di prova" [altra query ...]
"""
import sys
import numpy as np
import database
from models import gestore

def esegui():
    frames = database.carica_tutti_embedding_clip()
    if len(frames) < 5:
        print("Archivio troppo piccolo: importa ed elabora almeno 5 elementi.")
        return
    matrice = np.stack([f["embedding"] for f in frames])
    print(f"{len(frames)} frame con embedding.")

    # immagine->immagine: distribuzione dei coseni tra tutte le coppie
    coseni_img = (matrice @ matrice.T)[np.triu_indices(len(frames), k=1)]
    print(f"img->img   media={coseni_img.mean():.3f} sigma={coseni_img.std():.3f} "
          f"p95={np.percentile(coseni_img, 95):.3f}  (SOGLIA_SIMILARITA_IMMAGINE ~ p95)")

    qwen = gestore.ottieni_qwen()
    for query in sys.argv[1:] or ["una spiaggia al tramonto", "un documento con del testo"]:
        v = qwen.ottieni_embedding_testo(query)
        coseni = matrice @ v
        print(f"testo->img '{query}': media={coseni.mean():.3f} sigma={coseni.std():.3f} "
              f"max={coseni.max():.3f}")

if __name__ == "__main__":
    esegui()
```

- [ ] **Step 6: Run diagnostics**

Run: `..\venv\Scripts\python.exe test_models.py`
Expected: three OK lines. Then run all test scripts (`test_qwen_client.py`, `test_database_v07.py`, `test_persone.py`, `test_ricerca_vettoriale.py`) — all pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: rimozione CLIP/EasyOCR, diagnostica v0.7, script di calibrazione"
```

---

### Task 10: Docs + end-to-end verification + calibration

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md`** — model list (Qwen3-VL-Embedding-2B via llama-server, FaceNet, Whisper; no CLIP/EasyOCR), the "three search modes" note (now two Qwen scales + faces), the queue architecture (SQLite stage columns, `processor.avvia_lavoratore`), the Chroma collection rename (`qwen_frames`), the 2048-d dimension, `models/qwen/` requirement, and the new test file names.

- [ ] **Step 2: Update `README.md`** — file map (add `qwen_client.py`, `persone.py`, `calibra_soglie.py`; drop OCR mentions), the model prerequisites (`models/qwen/` contents), the Persone page, the queue pause/resume, the negative prompt.

- [ ] **Step 3: End-to-end verification (manual, the repo's real check):**
1. `.\scripts\windows\run.bat` — app starts, llama-server starts lazily on first AI need.
2. Import ~10 images + 1 short video from `model testing/testing images/`.
3. Upload returns in seconds; queue panel counts down; pause/resume works; restart the app mid-queue and confirm it resumes.
4. Text search (Italian query) returns sane results; negative prompt demotes matching content.
5. Image similarity with a query image works.
6. Persone page shows clustered faces; rename + merge work.
7. Video result plays at the frame timestamp; Whisper search finds spoken words.
8. Run `..\venv\Scripts\python.exe calibra_soglie.py` and adjust `SOGLIA_SIMILARITA_IMMAGINE` / `FATTORE_SIGMA_RICERCA_TESTO` / `SCALA_LOGIT_TAG` in config.py from the measured numbers.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: README e CLAUDE.md per v0.7 (Qwen, coda, persone)"
```

---

## Deviations from the spec (agreed rationale)

1. **llama-server lifecycle**: managed as a lazy subprocess by `models.GestoreQwen` (started on first embedding, stopped via `atexit`/`libera_memoria`) instead of by the run scripts. Fewer moving parts, cross-platform for free, matches the existing lazy-singleton pattern; run scripts stay untouched.
2. **Upload fast path**: upload = register only (hash/copy/DB row). Thumbnail/EXIF/frame-extraction moved into a fast "preparation" queue stage — extracting video frames means decoding the whole file, which is not "instant". The worker prioritizes preparation over AI stages so the gallery fills quickly.
3. **`ocr_text` column**: kept in the schema (dropped from UI and pipeline) — dropping a column buys nothing.

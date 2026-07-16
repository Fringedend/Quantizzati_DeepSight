# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DeepSight: a local Streamlit app to archive and search photos/videos with AI —
Qwen3-VL-Embedding-2B (semantic text/image search, image similarity, zero-shot tags;
served locally by a `llama-server` subprocess), MTCNN/FaceNet (face detection +
person clustering), Whisper (audio transcription). No CLIP, no EasyOCR — Qwen3-VL
replaced both. **All code identifiers, comments, and docs are in Italian** — match
that when editing.

## Commands

Launcher scripts live under `scripts/`, split per OS: `scripts/windows/` and
`scripts/linux/`. Nothing is in the repo root.

```powershell
# Install (one-time): creates venv, picks CUDA vs CPU PyTorch build, installs deps
.\scripts\windows\install.bat     # or: powershell -File scripts\windows\install.ps1

# Run the app (opens browser; Ctrl+C to stop)
.\scripts\windows\run.bat         # or: .\venv\Scripts\python.exe -m streamlit run src\app.py

# Unit check (assumption behind Chroma retrieval: L2 rank == cosine rank for normalized vectors)
.\venv\Scripts\python.exe src\test_ricerca_vettoriale.py

# Qwen client helpers (smart_resize, base64 encoding) — no llama-server needed
.\venv\Scripts\python.exe src\test_qwen_client.py

# Staged-queue + persons schema (temp DB, Chroma handles stubbed out — no real writes)
.\venv\Scripts\python.exe src\test_database.py

# Face clustering: greedy centroid assignment + DBSCAN re-cluster
.\venv\Scripts\python.exe src\test_persone.py

# Diagnostic: verify all three AI models load (Qwen via llama-server, FaceNet, Whisper)
.\venv\Scripts\python.exe src\test_models.py

# Manual (not a test): measure real Qwen cosine distributions on your own archive to
# (re)tune config.py's SCALA_LOGIT_TAG and inspect search-score distributions
.\venv\Scripts\python.exe src\calibra_soglie.py "una query di prova"
```

The `.bat` files exist only because double-clicking a `.ps1` opens it in an editor;
they invoke the `.ps1` sitting next to them.

Linux/macOS: same venv + CUDA-vs-CPU-PyTorch flow, but does **not** auto-install
Python (distros differ too much — apt/dnf/pacman) and detects the GPU via
`nvidia-smi` only, with no WMI fallback.

```bash
./scripts/linux/install.sh
./scripts/linux/run.sh
venv/bin/python src/test_ricerca_vettoriale.py
venv/bin/python src/test_qwen_client.py
venv/bin/python src/test_database.py
venv/bin/python src/test_persone.py
venv/bin/python src/test_models.py
```

There is no test framework. Tests are `assert`-based with an `if __name__ == "__main__"`
self-check — run the file directly.

## Non-obvious architecture

**Search = SQLite is the source of truth, Chroma is only an ANN index.** Embeddings
are stored twice: as float32 BLOBs in SQLite (`media_frames.clip_embedding` — the
column name is a holdover from the CLIP era, it now holds 2048-d Qwen vectors;
`faces.embedding`) and as vectors in ChromaDB. The search flow (`database.cerca_frame_simili`
/ `cerca_volti_simili`): Chroma returns the top-k *candidate ids*, then the **exact
cosine is recomputed in Python** (`np.dot`) against the SQLite BLOBs. Chroma's own
distance score is deliberately ignored — it defaults to L2, and the app's thresholds
(`config.SOGLIA_SIMILARITA_*`) are cosine. If `_store_frame`/`_store_volti` is `None`
(Chroma failed to load), both functions fall back to a full linear scan.

**Qwen searches have NO similarity threshold — only faces do.** Text→image and
image→image search both use Qwen3-VL-Embedding-2B (`models.GestoreQwen`), whose cosine
distributions sit on different, query-dependent scales — every fixed or adaptive cutoff
tried in practice either hid everything or nothing. So the UI (`app.py`, "Ricerca
Avanzata") never filters by score: results are sorted by relevance and paginated —
top 5, then a "Mostra altri" button (+10 per click; the counter resets via a signature
of the search inputs in `st.session_state["firma_ricerca"]`). Only face search keeps a
threshold (`SOGLIA_SIMILARITA_VOLTI` — FaceNet, empirically calibrated). `SCALA_LOGIT_TAG`
for zero-shot tags is still a CLIP-era placeholder; measure with `src/calibra_soglie.py`.
The text-search tab also supports an optional negative prompt: the UI embeds a second
"must not contain" phrase and the final score becomes
`cos(query, frame) - config.LAMBDA_PROMPT_NEGATIVO * cos(negative, frame)`.
`config.ISTRUZIONE_RICERCA` (the query-side retrieval instruction) is **deliberately in
Italian**: the instruction's language tells the model which language to expect the query
in — with an English instruction, the one-word query "Cane" embedded as English *cane*
(walking stick) and retrieved no dogs. Don't translate it back to English.

**Two separate Chroma collections (`qwen_frames`, `faces`) are mandatory, not cosmetic.**
`media_frames.id` and `faces.id` are independent AUTOINCREMENT sequences, so frame 1
and face 1 share id `"1"`. A single collection would collide and silently corrupt
retrieval (both vectors are 2048-d, so no dimension error surfaces it).

**Chroma self-heals from SQLite.** `database._sincronizza_indici_vettoriali()` runs
at the end of `inizializza_db()`: if a collection is empty but SQLite has rows, it
backfills from the BLOBs. So an existing archive stays searchable with no reimport,
and you can delete `chroma_db/` to force a rebuild.

**Imports are flat because Streamlit runs from `src/`.** Modules use `import config`,
`import database` (not `src.config`). `streamlit run src\app.py` puts `src/` on the
path. Any standalone script must run with `src/` as cwd or the imports break.

**Paths resolve one level up from `src/`.** `config.py` and `chroma_store.py` compute
`DIR_BASE`/`chroma_db` as `dirname(dirname(__file__))`, so `data/` and `chroma_db/`
always land at the project root regardless of where the process starts. `config.py`
creates all `data/` subfolders at import time.

**Models are lazy singletons, but Qwen is a subprocess, not an in-process model.**
`models.gestore` (a `GestoreModelli`) picks the device once (`config.MODALITA_DISPOSITIVO`:
auto/cpu/cuda, falling back to CPU if CUDA is absent) and loads FaceNet/Whisper only on
first use, same as before. `gestore.ottieni_qwen()` instead lazily spawns `llama-server`
(`models.GestoreQwen`, driven over HTTP by `src/qwen_client.py`) as a subprocess serving
`models/qwen/*.gguf` on `127.0.0.1:8091`; it's stopped by `libera_memoria()` or at process
exit (`atexit`). The sidebar's "Libera Memoria GPU" button calls `gestore.libera_memoria()`,
which also kills the llama-server subprocess. The `-ngl` flag (GPU offload) is no longer
hardcoded to 0: `ottieni_qwen` passes `config.QWEN_NGL` (default 99) when the detected device
is `cuda`, else 0, and `GestoreQwen.__init__` retries once with `-ngl 0` if the GPU start
fails (insufficient VRAM / incompatible CUDA driver). Actual GPU use also needs the CUDA
build of `llama-server` — the Windows installer fetches it when it detects an NVIDIA GPU with
≥4 GB VRAM; otherwise (CPU build, or Linux where llama.cpp ships no prebuilt CUDA) `-ngl>0`
is simply ignored and Qwen runs on CPU.

**Ingestion is a two-speed, resumable staged queue — not one synchronous pipeline.**
`processor.registra_file` does only the fast, AI-free part synchronously: dedup by
SHA-256, copy into `data/archive/` renamed to `<hash><ext>`, insert the `media_items` row
(`processed=0`). `processor.avvia_lavoratore()` starts (once per process) a daemon thread
that works off a queue that is *not* in memory — `database.prossimo_media_in_coda()` IS the
queue, so a restart resumes exactly where it left off. The thread first runs `_prepara_media`
(thumbnail, EXIF/GPS, video frame extraction — no AI, so the gallery fills quickly) which
flips `processed` to `1`, then the three independent AI stages, each tracked by its own
column on `media_items` (`stato_embedding`, `stato_volti`, `stato_trascrizione`; each
0=pending/1=done/-1=failed): `_stadio_embedding` (Qwen embedding + zero-shot tags via softmax
over `CATEGORIE`), `_stadio_volti` (MTCNN/FaceNet + assignment to a person via
`persone.assegna_persona`), `_stadio_trascrizione` (video only: ffmpeg audio extraction +
Whisper). `processor.stato_coda()` reports counts/ETA for the UI's queue panel. Pause/resume
(`metti_in_pausa()`/`riprendi()`) is a flag in the `impostazioni` key/value table
(`coda_in_pausa`), polled between stages so the in-flight item finishes cleanly;
`riprova_falliti()` resets any `-1` stage/item back to `0`. All tunable thresholds live in
`config.py`.

**Archive is app-managed; "intruder" files.** Because legit files are named by hash,
any file in `data/archive/` not in the DB was hand-copied and is unindexed. The
Dashboard's integrity panel and `processor.trova_file_intrusi` /
`importa_file_intrusi` / `sposta_file_intrusi_in_quarantena` detect and reconcile them.

## Dependencies

Both installers (`scripts/windows/install.ps1`, `scripts/linux/install.sh`) install
`torch`/`torchvision` first from the CUDA-or-CPU wheel index (auto-detected via
`nvidia-smi`, plus a WMI fallback on Windows), then `requirements.txt`, then
`facenet-pytorch` with `--no-deps` to prevent it from pulling NumPy < 2.0. **To add a
normal dependency, edit `requirements.txt` only** — neither script needs changes; keep
the two in sync when the install flow itself changes. FFmpeg comes from `imageio-ffmpeg`;
`models.py`/`processor.py` prepend its bin dir to `PATH` and feed Whisper a manually
decoded WAV array to avoid a Windows `ffmpeg.exe` lookup.

**Qwen's model files are auto-downloaded by the install scripts (~2.1 GB), not
committed** (GitHub rejects files >100 MB). Both installers end with an idempotent
download step: the two GGUFs from `DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF` on
Hugging Face, and `llama-server` (plus DLLs/libs) from the pinned llama.cpp
release `b10016`. On Windows the installer picks the **CUDA 12.4** build (llama zip +
`cudart` runtime zip, ~610 MB) when it detects an NVIDIA GPU with ≥4 GB VRAM, else the
**CPU** build (~17 MB); Linux always gets the CPU build (no prebuilt CUDA for Linux in
that release). Each GGUF is SHA-256-verified against the pinned hashes in the
scripts — a corrupted GGUF does NOT crash llama-server, it silently yields NaN
embeddings (happened in practice); on mismatch the file is deleted and re-running the
installer re-downloads it. On Windows the `.part`→final rename retries up to 5 times:
Defender locks big just-written files for a few seconds ("file in uso da un altro
processo"). `models/qwen/` must end up containing `llama-server`(`.exe` on
Windows), `Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf`, and
`mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` (paths in
`config.PERCORSO_LLAMA_SERVER`/`PERCORSO_MODELLO_QWEN`/`PERCORSO_MMPROJ_QWEN`).
Since Qwen loads lazily, a missing file doesn't surface at app startup — the embedding
stage fails per item (`stato_embedding = -1`); after installing the models, the queue
panel's "🔁 Riprova falliti" button re-queues them (it renders even when the queue is
otherwise empty — that empty-queue case is exactly the missing-models scenario).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DeepSight: a local Streamlit app to archive and search photos/videos with AI —
CLIP (semantic search + zero-shot tags), MTCNN/FaceNet (faces), EasyOCR (text in
images), Whisper (audio transcription). **All code identifiers, comments, and docs
are in Italian** — match that when editing.

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

# Diagnostic: verify all four AI models load
.\venv\Scripts\python.exe src\test_models.py
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
venv/bin/python src/test_models.py
```

There is no test framework. Tests are `assert`-based with an `if __name__ == "__main__"`
self-check — run the file directly.

## Non-obvious architecture

**Search = SQLite is the source of truth, Chroma is only an ANN index.** Embeddings
are stored twice: as float32 BLOBs in SQLite (`media_frames.clip_embedding`,
`faces.embedding`) and as vectors in ChromaDB. The search flow (`database.cerca_frame_simili`
/ `cerca_volti_simili`): Chroma returns the top-k *candidate ids*, then the **exact
cosine is recomputed in Python** (`np.dot`) against the SQLite BLOBs. Chroma's own
distance score is deliberately ignored — it defaults to L2, and the app's thresholds
(`config.SOGLIA_SIMILARITA_*`) are cosine. If `_store_frame`/`_store_volti` is `None`
(Chroma failed to load), both functions fall back to a full linear scan.

**The three search modes live on different cosine scales — no shared threshold.** Text→image
(multilingual encoder) cosines sit in a narrow band (~0.13–0.27) whose noise floor shifts per
query, so there is *no* usable constant: `database.soglia_adattiva_testo` recomputes the cutoff
per query as `media + FATTORE_SIGMA_RICERCA_TESTO * sigma` over **all** frame embeddings (not the
Chroma candidates — those are the top ones, so their mean is inflated). Image→image cosines are
far higher (median ~0.48), and use their own constant `SOGLIA_SIMILARITA_IMMAGINE`; faces use
`SOGLIA_SIMILARITA_VOLTI`. Reusing one threshold across modes is what made image search return
the entire archive.

**Two separate Chroma collections (`clip_frames`, `faces`) are mandatory, not cosmetic.**
`media_frames.id` and `faces.id` are independent AUTOINCREMENT sequences, so frame 1
and face 1 share id `"1"`. A single collection would collide and silently corrupt
retrieval (both vectors are 512-d, so no dimension error surfaces it).

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

**Models are lazy singletons.** `models.gestore` (a `GestoreModelli`) picks the device
once (`config.MODALITA_DISPOSITIVO`: auto/cpu/cuda, falling back to CPU if CUDA is
absent) and loads each of CLIP/FaceNet/EasyOCR/Whisper only on first use. The
sidebar's "Libera Memoria GPU" button calls `gestore.libera_memoria()`.

**Ingestion pipeline** (`processor.aggiungi_e_elabora_file`): dedup by SHA-256 →
copy into `data/archive/` renamed to `<hash><ext>` → insert row (`processed=0`) →
per media type run `elabora_immagine`/`elabora_video` (EXIF/GPS, CLIP embedding,
zero-shot tags via softmax over `CATEGORIE`, OCR, face crops; videos also sample
frames every `INTERVALLO_FRAME_VIDEO`s and Whisper-transcribe) → `processed=1`
(or `-1` on failure). All tunable thresholds live in `config.py`.

**Archive is app-managed; "intruder" files.** Because legit files are named by hash,
any file in `data/archive/` not in the DB was hand-copied and is unindexed. The
Dashboard's integrity panel and `processor.trova_file_intrusi` /
`importa_file_intrusi` / `sposta_file_intrusi_in_quarantena` detect and reconcile them.

## Dependencies

Both installers (`scripts/windows/install.ps1`, `scripts/linux/install.sh`) install
`torch`/`torchvision` first from the CUDA-or-CPU wheel index (auto-detected via
`nvidia-smi`, plus a WMI fallback on Windows), then `requirements.txt`, then
`facenet-pytorch` and `easyocr` with `--no-deps` to prevent them from pulling
NumPy < 2.0. **To add a normal dependency, edit `requirements.txt` only** — neither
script needs changes; keep the two in sync when the install flow itself changes. FFmpeg
comes from `imageio-ffmpeg`; `models.py`/`processor.py` prepend its bin dir to `PATH`
and feed Whisper a manually decoded WAV array to avoid a Windows `ffmpeg.exe` lookup.

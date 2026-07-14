# DeepSight v0.7 — Qwen3-VL Embedding Integration — Design

**Date:** 2026-07-14
**Status:** Approved by user (planning phase)

Decisions locked during brainstorming:

| Decision | Choice |
|---|---|
| Embedding model | **Qwen3-VL-Embedding-2B only** (Q5_K_M GGUF, 2048-d) |
| Runtime | **llama-server subprocess** (HTTP `/embeddings`) |
| Face model | **Keep MTCNN + FaceNet** (already integrated) |
| Queue scope | **All AI work** in one pausable background queue |

Rationale for 2B-only: 2B (2048-d) and 8B (4096-d) embeddings live in different
vector spaces and dual-tower retrieval requires query and corpus embedded by the
same model — "both" would mean two full indexes at 19s/image. 2B scores 73.2 vs
8B's 77.8 on MMEB-V2, runs 2.8× faster (5s vs 14s per image on the target CPU),
and the assignment mandates no-GPU operation.

## 1. Overall approach: evolve in place

Modify the existing codebase; no rewrite. SQLite stays the source of truth,
ChromaDB stays the ANN index (candidate ids only, exact cosine recomputed in
Python), lazy model singletons in `models.gestore`, flat `src/` imports.

Changes: CLIP → Qwen3-VL-Embedding-2B; EasyOCR deleted; three new features
(negative prompt, face database, background queue).

## 2. Model layer

- **llama-server subprocess.** The run script starts `llama-server.exe` with the
  2B GGUF + its `mmproj` file before Streamlit, and stops it on exit.
  `models.py` replaces the CLIP loader with a small HTTP client for
  `/embeddings` (text, image, and image-with-text inputs). Server unreachable →
  clear user-facing error in search and indexing; no fallback model.
- **EasyOCR removed.** Text-in-image retrieval rides on Qwen embeddings (its OCR
  understanding was validated in model testing). The `testo_ocr` column, its
  extraction step, and its UI are deleted.
- **Zero-shot tags kept, re-based on Qwen.** Category tags = cosine of image
  embedding vs. embedded `CATEGORIE` names (text vectors computed once, cached).
- **Whisper and MTCNN/FaceNet unchanged.**
- **No reranker.** A (query, image) pair through Qwen3-VL-Reranker is a full VL
  forward pass (~5s CPU) → ~50s to rerank top-10. Cosine ranking already covers
  the assignment's optional ranking requirement.
- **Breaking change:** embeddings 512-d → 2048-d. Existing archives are marked
  embedding-pending and reindexed through the queue. New Chroma collection for
  the new dimension; the self-heal backfill from SQLite BLOBs keeps working.
- **Video:** frames are extracted at upload (existing `INTERVALLO_FRAME_VIDEO`
  sampling) and each frame is embedded as a static image — per the Qwen spec,
  video frames use the same `smart_resize` factor-32 rule as images, so no
  separate video path.

## 3. Search

- **Text search:** embed query with a retrieval instruction, cosine over frame
  embeddings. The adaptive threshold mechanism
  (`media + FATTORE_SIGMA_RICERCA_TESTO · sigma` over all frame embeddings)
  stays, but its constants were tuned to CLIP's ~0.13–0.27 cosine band —
  **they must be re-measured** on real data after switching to Qwen. Same for
  `SOGLIA_SIMILARITA_IMMAGINE`.
- **Negative prompt:** toggle + second text field in text search. Final score =
  `cos(query, img) − λ · cos(negative, img)`, λ in `config.py` (start 0.5).
  Implemented as re-scoring of the candidate set: costs one extra text embedding
  per search. Hard-filter variant deferred until the penalty proves too soft.
- **Image similarity:** query image → Qwen embedding → same index, own constant
  threshold (as today).
- Qwen supports 33 languages including Italian natively — expected improvement
  for Italian queries over multilingual CLIP.

## 4. Face Database ("Persone")

- Schema: new `persons` table (`id`, `name`), new `faces.person_id` FK.
- **Clustering:** incremental greedy assignment — a new face joins the closest
  existing person if cosine to that person's centroid ≥ threshold
  (new `config.py` constant), else it founds a new unnamed person. A
  "re-cluster all" action (DBSCAN, sklearn already installed) rebuilds
  assignments when the threshold was mis-tuned.
  <!-- ponytail: greedy centroid assignment; upgrade to full re-cluster/HAC if merge errors annoy users -->
- **UI:** "Persone" section — grid of persons (representative crop + media
  count); click a person → all images/videos containing them; inline rename;
  select-two + merge button (reassigns `person_id`, recomputes centroid).

## 5. Background queue

- **State in SQLite, not a queue table.** Three stage columns on the media row:
  `stato_embedding`, `stato_volti`, `stato_trascrizione`
  (0 = pending, 1 = done, −1 = failed). The queue is the query "any stage
  pending, oldest first". Crash-safe by construction: restart resumes where it
  stopped.
- **Upload fast path (synchronous):** hash → dedup → copy to archive → EXIF/GPS
  → thumbnail → DB row → (video) frame extraction. All AI stages start pending.
  10,000-photo upload completes in minutes.
- **Worker:** one daemon thread in the Streamlit process, one file at a time,
  checks a `paused` flag (settings table) between files. Pause finishes the
  current file, then idles — no mid-inference cancellation.
  <!-- ponytail: single worker thread; separate worker process only if Streamlit reruns interfere -->
- **UI:** dashboard panel — pending count, current file, Pause/Resume, ETA from
  rolling average seconds/item.

## 6. Config & scripts

- New `config.py` entries: GGUF and mmproj paths, llama-server host/port/thread
  count, negative-prompt λ, face-clustering threshold, recalibrated search
  thresholds.
- Install script: place/download GGUF + mmproj + llama-server binaries.
- Run script: launch llama-server, wait for readiness, launch Streamlit, kill
  the server on exit. (Windows and Linux variants, as today.)
- Satisfies the assignment's "config in a text file" and "install/run scripts"
  requirements with existing mechanisms.

## 7. Error handling

- llama-server down: search shows an error banner; queue worker idles and
  retries instead of marking files failed.
- Per-file stage failure: stage set to −1, surfaced in the existing integrity
  panel with a retry action.

## 8. Testing

Assert-based self-check scripts, run directly (repo convention, no framework):

1. Embedding client returns a normalized 2048-d vector for a text and an image
   input (requires llama-server running).
2. Negative prompt demotes a known-matching image below a neutral one.
3. Greedy clustering groups two crops of the same test person and separates a
   different person.

## Out of scope (deliberately)

- Qwen3-VL-Reranker (CPU-infeasible per search, see §2).
- 8B model support / model switching.
- Hard-filter negative prompt (revisit after λ-penalty is evaluated).
- Replacing FaceNet with InsightFace (revisit only if clustering quality
  disappoints).
- Mid-file pause/cancel in the queue.

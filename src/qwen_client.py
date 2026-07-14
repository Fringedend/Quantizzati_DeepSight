"""Client llama-server + preprocessing immagini per Qwen3-VL-Embedding.

Nota: questo file e' vendored (copiato quasi verbatim) da un progetto benchmark
gia' testato ("model testing/benchmark/preprocess.py" + "embed.py"). Per
fedelta' al codice sorgente originale, i commenti e i docstring interni
restano in inglese: unica eccezione sanzionata alla regola "solo italiano"
del repository.

Confirmed working recipe (llama.cpp main, 2026-07):

    LLAMA_MEDIA_MARKER=<__media__> llama-server -m MODEL --mmproj PROJ \
        --embedding --pooling last --embd-normalize 2 --host H --port P

  * --pooling last     -> last-token (EOS) hidden state = the README's representation
  * --embd-normalize 2 -> L2-normalized output vectors (so dot product == cosine)

Client posts to /embeddings with a `content` array; images ride along as base64 in
`multimodal_data` next to a `<__media__>` marker in `prompt_string`.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

MEDIA_MARKER = "<__media__>"


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 4096,
    max_pixels: int = 1843200,
) -> tuple[int, int]:
    """Return (new_height, new_width), each a multiple of `factor`.

    Mirrors the Qwen2/3-VL smart_resize logic. Keeps the aspect ratio as close as
    possible while staying within [min_pixels, max_pixels] total pixels and making
    both sides divisible by `factor`.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


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


class EmbedError(RuntimeError):
    pass


class LlamaServer:
    """Context manager that launches one llama-server for a single model and tears it down."""

    def __init__(
        self,
        exe: str | Path,
        model_path: str | Path,
        mmproj_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 8077,
        threads: int = 0,
        context: int = 8192,
        startup_timeout_s: int = 240,
    ):
        self.exe = str(exe)
        self.model_path = str(model_path)
        self.mmproj_path = str(mmproj_path)
        self.host = host
        self.port = port
        self.threads = threads
        self.context = context
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None
        self._log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _cmd(self) -> list[str]:
        cmd = [
            self.exe,
            "-m", self.model_path,
            "--mmproj", self.mmproj_path,
            "--embedding",
            "--pooling", "last",
            "--embd-normalize", "2",
            "-c", str(self.context),
            "-ngl", "0",              # force CPU: no GPU layers
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.threads and self.threads > 0:
            cmd += ["--threads", str(self.threads)]
        return cmd

    def avvia(self) -> "LlamaServer":
        env = dict(os.environ)
        env["LLAMA_MEDIA_MARKER"] = MEDIA_MARKER
        import config
        log_dir = Path(config.DIR_DATI) / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"server-{Path(self.model_path).stem}.log"
        self._log_fh = open(self._log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            self._cmd(), env=env, stdout=self._log_fh, stderr=subprocess.STDOUT
        )
        try:
            self.wait_healthy(self.startup_timeout_s)
        except Exception:
            self.ferma()
            raise
        return self

    def ferma(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        if getattr(self, "_log_fh", None):
            self._log_fh.close()
            self._log_fh = None

    def wait_healthy(self, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        url = self.base_url + "/health"
        last_err = None
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise EmbedError(
                    f"llama-server exited early (code {self.proc.returncode}). "
                    f"See log: {self._log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        body = json.loads(r.read() or b"{}")
                        if body.get("status", "ok") in ("ok", "loading model") and \
                                body.get("status") != "loading model":
                            return
                        if body.get("status") == "ok":
                            return
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(1.0)
        raise EmbedError(
            f"llama-server did not become healthy within {timeout_s}s "
            f"(last error: {last_err}). See log: {self._log_path}"
        )

    # -- embedding calls -------------------------------------------------------

    def _post_embeddings(self, content: list[dict]) -> object:
        payload = json.dumps({"content": content}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise EmbedError(f"/embeddings HTTP {e.code}: {detail}") from e

    @staticmethod
    def _extract_vector(resp: object) -> np.ndarray:
        """Pull a single 1-D embedding vector out of whatever shape the server returns."""
        node = resp
        if isinstance(node, dict):
            if "data" in node:            # OpenAI-style
                node = node["data"]
            elif "embedding" in node:
                node = node["embedding"]
        if isinstance(node, list) and node and isinstance(node[0], dict):
            node = node[0].get("embedding", node[0].get("data"))
        arr = np.asarray(node, dtype=np.float32)
        if arr.ndim == 2:                 # per-token matrix -> take last token (EOS)
            arr = arr[-1]
        elif arr.ndim != 1:
            arr = arr.reshape(-1)
        n = np.linalg.norm(arr)
        return arr / n if n > 0 else arr

    def embed_text(self, text: str, instruction: str | None = None) -> np.ndarray:
        """Embed a text query. Qwen3-VL-Embedding is instruction-aware: for retrieval
        queries, wrap the text as `Instruct: <task>\\nQuery: <text>` (documents/images
        get NO instruction). Passing instruction=None embeds the raw text.
        """
        prompt = f"Instruct: {instruction}\nQuery: {text}" if instruction else text
        return self._extract_vector(self._post_embeddings([{"prompt_string": prompt}]))

    def embed_image_b64(self, b64: str) -> np.ndarray:
        content = [{"prompt_string": MEDIA_MARKER, "multimodal_data": [b64]}]
        return self._extract_vector(self._post_embeddings(content))

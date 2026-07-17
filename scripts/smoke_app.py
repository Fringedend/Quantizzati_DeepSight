#!/usr/bin/env python3
"""Avvia Streamlit con dati temporanei, verifica /health e lo chiude."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="deepsight-smoke-") as cartella:
        base = Path(cartella)
        ambiente = os.environ.copy()
        ambiente["DEEPSIGHT_DATA_DIR"] = str(base / "data")
        ambiente["DEEPSIGHT_CHROMA_DIR"] = str(base / "chroma")
        comando = [
            sys.executable, "-m", "streamlit", "run", "src/app.py",
            "--server.headless=true", "--server.port=8765",
            "--browser.gatherUsageStats=false",
        ]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        processo = subprocess.Popen(
            comando, cwd=RADICE, env=ambiente,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, creationflags=flags,
        )
        try:
            for _ in range(40):
                if processo.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                            "http://127.0.0.1:8765/_stcore/health", timeout=1) as risposta:
                        if risposta.status == 200:
                            print("Streamlit health check: OK")
                            return 0
                except Exception:
                    time.sleep(0.5)
            uscita = processo.stdout.read() if processo.stdout else ""
            print("Streamlit smoke test fallito:\n" + uscita, file=sys.stderr)
            return 1
        finally:
            if processo.poll() is None:
                processo.terminate()
                try:
                    processo.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    processo.kill()
                    processo.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())


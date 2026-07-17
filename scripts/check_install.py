#!/usr/bin/env python3
"""Diagnostica leggera post-installazione, senza caricare i modelli AI."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

import config  # noqa: E402


MODULI = (
    "numpy", "PIL", "cv2", "streamlit", "chromadb", "sklearn",
    "reverse_geocoder", "torch", "torchvision", "whisper",
    "imageio_ffmpeg", "facenet_pytorch",
)


def main() -> int:
    errori = []
    print(f"Python: {sys.version.split()[0]} ({sys.platform})")
    for nome in MODULI:
        try:
            modulo = importlib.import_module(nome)
            versione = getattr(modulo, "__version__", "ok")
            print(f"  OK {nome}: {versione}")
        except Exception as errore:
            errori.append(f"{nome}: {errore}")
            print(f"  ERRORE {nome}: {errore}")

    try:
        with tempfile.NamedTemporaryFile(dir=config.DIR_DATI):
            pass
        print(f"  OK cartella dati scrivibile: {config.DIR_DATI}")
    except Exception as errore:
        errori.append(f"cartella dati: {errore}")

    for percorso in (config.PERCORSO_LLAMA_SERVER, config.PERCORSO_MODELLO_QWEN,
                      config.PERCORSO_MMPROJ_QWEN):
        if not os.path.isfile(percorso):
            errori.append(f"file Qwen mancante: {percorso}")
        else:
            print(f"  OK {os.path.basename(percorso)}")

    if os.name != "nt" and os.path.isfile(config.PERCORSO_LLAMA_SERVER) and not os.access(
            config.PERCORSO_LLAMA_SERVER, os.X_OK):
        errori.append("llama-server non è eseguibile")

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if not os.path.isfile(ffmpeg):
            raise FileNotFoundError(ffmpeg)
        print(f"  OK ffmpeg: {ffmpeg}")
    except Exception as errore:
        errori.append(f"ffmpeg: {errore}")

    if errori:
        print("\nDiagnostica fallita:")
        for errore in errori:
            print(f"  - {errore}")
        return 1
    print("\nInstallazione DeepSight valida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


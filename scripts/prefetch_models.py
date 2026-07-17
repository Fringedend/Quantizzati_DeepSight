#!/usr/bin/env python3
"""Scarica in models/ i pesi FaceNet e Whisper senza elaborare contenuti utente."""

from __future__ import annotations

import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

import config  # noqa: E402


def main() -> int:
    print("Scaricamento FaceNet (vggface2)...")
    from facenet_pytorch import InceptionResnetV1
    modello_volti = InceptionResnetV1(pretrained="vggface2")
    del modello_volti

    print(f"Scaricamento Whisper '{config.NOME_MODELLO_WHISPER}'...")
    import whisper
    modello_whisper = whisper.load_model(
        config.NOME_MODELLO_WHISPER,
        device="cpu",
        download_root=config.DIR_MODELLI_WHISPER,
    )
    del modello_whisper
    print("Modelli FaceNet e Whisper disponibili localmente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


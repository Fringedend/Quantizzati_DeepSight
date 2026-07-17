#!/usr/bin/env python3
"""Esegue i test leggeri in processi isolati su Windows e Linux."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
TEST = (
    "test_path_utils.py",
    "test_path_migration.py",
    "test_database.py",
    "test_gallery_utils.py",
    "test_persone.py",
    "test_qwen_client.py",
    "test_ricerca_vettoriale.py",
)


def main() -> int:
    for nome in TEST:
        percorso = RADICE / "src" / nome
        print(f"\n== {nome} ==", flush=True)
        completato = subprocess.run([sys.executable, str(percorso)], cwd=RADICE, check=False)
        if completato.returncode:
            return completato.returncode
    print("\nTutti i test leggeri sono passati.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


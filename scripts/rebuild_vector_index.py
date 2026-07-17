#!/usr/bin/env python3
"""Riallinea gli indici Chroma agli embedding conservati in SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

import database  # noqa: E402


def main() -> int:
    database.inizializza_db()
    esito = database.sincronizza_indici_vettoriali(forza=True)
    if esito.get("errore"):
        print(f"ERRORE: {esito['errore']}", file=sys.stderr)
        return 1
    if esito["frame"] is None or esito["volti"] is None:
        print("ERRORE: ChromaDB non è disponibile; controlla l'installazione.", file=sys.stderr)
        return 1
    print(f"Frame indicizzati: {esito['frame'] or 0}")
    print(f"Volti indicizzati: {esito['volti'] or 0}")
    print(f"Vettori orfani rimossi: {esito['rimossi_frame'] + esito['rimossi_volti']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

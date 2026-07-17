#!/usr/bin/env python3
"""Analizza o converte i percorsi di archive.db al formato portabile."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

import config  # noqa: E402
from path_migration import (  # noqa: E402
    analizza_percorsi, applica_migrazione, crea_backup_sqlite,
    trova_percorsi_non_portabili,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Converte i percorsi assoluti Windows/Linux in percorsi relativi al progetto."
    )
    gruppo = parser.add_mutually_exclusive_group()
    gruppo.add_argument("--apply", action="store_true", help="applica davvero le modifiche")
    gruppo.add_argument("--dry-run", action="store_true", help="mostra soltanto le modifiche (default)")
    parser.add_argument("--db", default=config.PERCORSO_DB, help="database SQLite da analizzare")
    args = parser.parse_args()

    percorso_db = Path(args.db).resolve()
    if not percorso_db.is_file():
        print(f"ERRORE: database non trovato: {percorso_db}", file=sys.stderr)
        return 2

    connessione = sqlite3.connect(percorso_db)
    try:
        modifiche = analizza_percorsi(connessione, base_dir=RADICE)
        irrisolti = trova_percorsi_non_portabili(connessione, base_dir=RADICE)
        print(f"Percorsi da convertire: {len(modifiche)}")
        for modifica in modifiche[:20]:
            print(f"  {modifica['tabella']}#{modifica['id']}: "
                  f"{modifica['prima']} -> {modifica['dopo']}")
        if len(modifiche) > 20:
            print(f"  ... e altri {len(modifiche) - 20}")
        if irrisolti:
            print(f"Percorsi esterni/non riconosciuti da verificare manualmente: {len(irrisolti)}")
            for elemento in irrisolti[:20]:
                print(f"  {elemento['tabella']}#{elemento['id']}: {elemento['valore']}")

        if not args.apply:
            print("Nessuna modifica applicata. Usa --apply dopo aver controllato l'anteprima.")
            return 0
        if not modifiche:
            print("Database già portabile: niente da fare.")
            return 0

        suffisso = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = percorso_db.with_name(f"{percorso_db.name}.backup-{suffisso}")
        crea_backup_sqlite(connessione, backup)
        aggiornati = applica_migrazione(connessione, modifiche)
        print(f"Migrazione completata: {aggiornati} percorsi aggiornati.")
        print(f"Backup recuperabile: {backup}")
        return 1 if irrisolti else 0
    except Exception as errore:
        print(f"ERRORE: migrazione annullata: {errore}", file=sys.stderr)
        return 1
    finally:
        connessione.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Migrazione idempotente dei percorsi SQLite al formato relativo portabile."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import config
from path_utils import percorso_da_salvare, percorso_e_portabile


COLONNE_PERCORSO = (
    ("media_items", "id", "file_path"),
    ("media_frames", "id", "image_path"),
    ("faces", "id", "crop_path"),
)


def analizza_percorsi(connessione: sqlite3.Connection, base_dir=None) -> list[dict]:
    """Elenca le modifiche necessarie senza scrivere nel database."""
    modifiche = []
    for tabella, chiave, colonna in COLONNE_PERCORSO:
        esiste = connessione.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (tabella,)
        ).fetchone()
        if not esiste:
            continue
        for id_riga, valore in connessione.execute(
            f"SELECT {chiave}, {colonna} FROM {tabella} WHERE {colonna} IS NOT NULL"
        ):
            nuovo = percorso_da_salvare(valore, base_dir=base_dir)
            if nuovo != valore:
                modifiche.append({
                    "tabella": tabella, "chiave": chiave, "id": id_riga,
                    "colonna": colonna, "prima": valore, "dopo": nuovo,
                })
    return modifiche


def trova_percorsi_non_portabili(connessione: sqlite3.Connection, base_dir=None) -> list[dict]:
    """Elenca assoluti esterni/non riconoscibili che richiedono verifica umana."""
    irrisolti = []
    for tabella, chiave, colonna in COLONNE_PERCORSO:
        esiste = connessione.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (tabella,)
        ).fetchone()
        if not esiste:
            continue
        for id_riga, valore in connessione.execute(
            f"SELECT {chiave}, {colonna} FROM {tabella} WHERE {colonna} IS NOT NULL"
        ):
            if (not percorso_e_portabile(valore)
                    and percorso_da_salvare(valore, base_dir=base_dir) == valore):
                irrisolti.append({
                    "tabella": tabella, "id": id_riga, "colonna": colonna, "valore": valore,
                })
    return irrisolti


def crea_backup_sqlite(connessione: sqlite3.Connection, percorso_backup) -> str:
    """Crea un backup coerente anche quando il DB usa WAL."""
    destinazione = Path(percorso_backup)
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    backup = sqlite3.connect(destinazione)
    try:
        connessione.backup(backup)
    finally:
        backup.close()
    return str(destinazione)


def applica_migrazione(connessione: sqlite3.Connection, modifiche: list[dict]) -> int:
    """Applica tutte le modifiche in un'unica transazione atomica."""
    if not modifiche:
        return 0
    try:
        connessione.execute("BEGIN")
        for modifica in modifiche:
            connessione.execute(
                f"UPDATE {modifica['tabella']} SET {modifica['colonna']} = ? "
                f"WHERE {modifica['chiave']} = ?",
                (modifica["dopo"], modifica["id"]),
            )
        connessione.commit()
    except Exception:
        connessione.rollback()
        raise
    return len(modifiche)


def migra_percorsi_database(connessione: sqlite3.Connection, crea_backup=True, base_dir=None) -> dict:
    """Analizza, protegge e converte i percorsi; ritorna un riepilogo."""
    modifiche = analizza_percorsi(connessione, base_dir=base_dir)
    irrisolti = trova_percorsi_non_portabili(connessione, base_dir=base_dir)
    backup = None
    if modifiche and crea_backup:
        connessione.commit()
        backup = f"{config.PERCORSO_DB}.pre-percorsi.bak"
        if not os.path.exists(backup):
            crea_backup_sqlite(connessione, backup)
    aggiornati = applica_migrazione(connessione, modifiche)
    return {"analizzati": len(modifiche), "aggiornati": aggiornati,
            "irrisolti": len(irrisolti), "backup": backup}

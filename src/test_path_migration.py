"""Test dry-run, atomicità e idempotenza della migrazione percorsi."""

import sqlite3
import tempfile
from pathlib import Path

from path_migration import (analizza_percorsi, applica_migrazione, crea_backup_sqlite,
                            trova_percorsi_non_portabili)


def esegui_test():
    with tempfile.TemporaryDirectory() as cartella:
        radice = Path(cartella) / "DeepSight"
        radice.mkdir()
        db = Path(cartella) / "archivio.db"
        connessione = sqlite3.connect(db)
        connessione.executescript("""
            CREATE TABLE media_items (id INTEGER PRIMARY KEY, file_path TEXT UNIQUE);
            CREATE TABLE media_frames (id INTEGER PRIMARY KEY, image_path TEXT);
            CREATE TABLE faces (id INTEGER PRIMARY KEY, crop_path TEXT);
        """)
        connessione.execute("INSERT INTO media_items VALUES (1, ?)",
                            (r"C:\\Old\\DeepSight\\data\\archive\\a.jpg",))
        percorso_esterno = str((Path(cartella) / "sorgente-esterna.jpg").resolve())
        connessione.execute("INSERT INTO media_items VALUES (2, ?)", (percorso_esterno,))
        connessione.execute("INSERT INTO media_frames VALUES (1, ?)",
                            (r"C:\\Old\\DeepSight\\data\\frames\\f.jpg",))
        connessione.execute("INSERT INTO faces VALUES (1, ?)",
                            (r"C:\\Old\\DeepSight\\data\\faces\\v.jpg",))
        connessione.commit()

        modifiche = analizza_percorsi(connessione, base_dir=radice)
        assert len(modifiche) == 3
        irrisolti = trova_percorsi_non_portabili(connessione, base_dir=radice)
        assert len(irrisolti) == 1 and irrisolti[0]["valore"] == percorso_esterno
        backup = Path(cartella) / "backup.db"
        crea_backup_sqlite(connessione, backup)
        assert backup.is_file()
        assert applica_migrazione(connessione, modifiche) == 3
        assert connessione.execute("SELECT file_path FROM media_items").fetchone()[0] == \
            "data/archive/a.jpg"
        assert analizza_percorsi(connessione, base_dir=radice) == []
        connessione.close()
    print("test_path_migration: OK")


if __name__ == "__main__":
    esegui_test()

"""Ricalcola i tag zero-shot di TUTTI i frame in archivio.

Da lanciare dopo una modifica a processor.CATEGORIE: riusa gli embedding
gia' salvati in SQLite (nessuna immagine viene ri-embeddata), quindi il costo
e' solo l'embedding una tantum delle categorie via llama-server + prodotti
scalari. Eseguire dalla radice del progetto:

    Windows: .\\venv\\Scripts\\python.exe src\\ricalcola_tag.py
    Linux:   ./venv/bin/python src/ricalcola_tag.py
"""
import json

import database
import processor
from models import gestore


def main():
    frames = database.carica_tutti_embedding_clip()
    if not frames:
        print("Nessun frame con embedding in archivio: niente da fare.")
        return

    connessione = database.ottieni_connessione()
    campione = []
    try:
        for frame in frames:
            tags = processor.classifica_tag(frame["embedding"])
            # Solo la colonna dei tag: gli embedding non cambiano, Chroma non va toccata.
            connessione.execute("UPDATE media_frames SET objects = ? WHERE id = ?",
                                (json.dumps(tags), frame["frame_id"]))
            if len(campione) < 5:
                campione.append((frame["filename"], tags))
        connessione.commit()
    except Exception:
        connessione.rollback()
        raise
    finally:
        connessione.close()

    print(f"Tag ricalcolati per {len(frames)} frame.")
    print("\nCampione (primi 5):")
    for nome, tags in campione:
        print(f"  {nome}: {', '.join(tags) if tags else '(nessun tag)'}")

    gestore.libera_memoria()


if __name__ == "__main__":
    main()

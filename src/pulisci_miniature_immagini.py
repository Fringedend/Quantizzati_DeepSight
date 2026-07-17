"""Rimuove le miniature delle IMMAGINI ormai inutilizzate.

Da quando galleria/dashboard/ricerca mostrano l'originale ridimensionato al volo,
le immagini non usano più la miniatura da 300px (vedi processor.crea_anteprima,
che ormai la genera solo per i video). Le miniature-immagine già presenti su disco
sono quindi peso morto: questo script le elimina, lasciando intatte quelle dei video
(che servono ancora come fotogramma di ripiego).

Le miniature sono salvate col nome-hash del file archiviato (basename di file_path),
quindi per ogni immagine in archivio si cancella <hash>.jpg da config.DIR_ANTEPRIME.

Eseguire con src/ come cwd:
    .\\venv\\Scripts\\python.exe src\\pulisci_miniature_immagini.py
"""
import os

import config
import database


def pulisci_miniature_immagini():
    """Elimina le miniature delle immagini. Ritorna (rimosse, spazio_liberato_byte)."""
    connessione = database.ottieni_connessione()
    righe = connessione.execute(
        "SELECT file_path FROM media_items WHERE media_type = 'image'"
    ).fetchall()
    connessione.close()

    rimosse, spazio_liberato = 0, 0
    for riga in righe:
        # ottieni_connessione non imposta sqlite3.Row: le righe sono tuple posizionali.
        nome_senza_est, _ = os.path.splitext(os.path.basename(riga[0]))
        percorso_miniatura = os.path.join(config.DIR_ANTEPRIME, f"{nome_senza_est}.jpg")
        if os.path.exists(percorso_miniatura):
            try:
                spazio_liberato += os.path.getsize(percorso_miniatura)
                os.remove(percorso_miniatura)
                rimosse += 1
            except OSError as errore:
                print(f"Impossibile rimuovere {percorso_miniatura}: {errore}")

    return rimosse, spazio_liberato


if __name__ == "__main__":
    # Solo lettura di media_items + rimozione file su disco: nessuna scrittura sul DB
    # e nessun accesso a Chroma, così è sicuro anche con l'app in esecuzione (WAL).
    rimosse, spazio_liberato = pulisci_miniature_immagini()
    print(
        f"Miniature immagine rimosse: {rimosse} "
        f"({spazio_liberato / 1024:.1f} KB liberati). "
        "Le miniature dei video sono state lasciate intatte."
    )

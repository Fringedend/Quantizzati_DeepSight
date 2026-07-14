"""Migrazione v0.7: colonne di stadio, persone, impostazioni, coda."""
import os, tempfile, numpy as np

# DB temporaneo PRIMA di importare i moduli (config crea le cartelle al primo import)
os.environ.setdefault("DEEPSIGHT_TEST", "1")
import config
config.PERCORSO_DB = os.path.join(tempfile.mkdtemp(), "test.db")
import database
# Il percorso di ChromaDB NON e' configurabile: si azzerano gli handle per non
# scrivere vettori di prova nell'indice reale (tutti i chiamanti hanno il fallback).
database._store_frame = None
database._store_volti = None

def esegui_test():
    database.inizializza_db()
    # impostazioni
    assert database.leggi_impostazione("coda_in_pausa", "0") == "0"
    database.scrivi_impostazione("coda_in_pausa", "1")
    assert database.leggi_impostazione("coda_in_pausa") == "1"

    # media + frame senza embedding, poi aggiornato
    id_media = database.aggiungi_elemento_multimediale("/tmp/x.jpg", "x.jpg", "image", 1, "hash1")
    id_frame = database.crea_frame(id_media, 0, 0.0, "/tmp/x.jpg")
    frames = database.ottieni_frame_di_media(id_media)
    assert len(frames) == 1 and frames[0]["clip_embedding_presente"] is False
    emb = np.ones(config.DIM_EMBEDDING_QWEN, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    database.aggiorna_embedding_frame(id_frame, id_media, emb, ["cat"])
    assert database.ottieni_frame_di_media(id_media)[0]["clip_embedding_presente"] is True

    # coda: media appena registrato (processed=0) -> preparazione
    lavoro = database.prossimo_media_in_coda()
    assert lavoro is not None and lavoro["id"] == id_media and lavoro["processed"] == 0
    database.aggiorna_stato_elaborazione(id_media=id_media, stato_elaborazione=1)
    lavoro = database.prossimo_media_in_coda()  # ora tocca agli stadi AI
    assert lavoro["id"] == id_media
    for col in ("stato_embedding", "stato_volti", "stato_trascrizione"):
        database.imposta_stato_stadio(id_media, col, 1)
    assert database.prossimo_media_in_coda() is None
    conteggi = database.conteggio_coda()
    assert conteggi["da_preparare"] == 0 and conteggi["embedding"] == 0

    # persone
    id_p = database.crea_persona()
    id_volto = database.aggiungi_volto(id_media, id_frame, "/tmp/f.jpg",
                                       np.ones(512, dtype=np.float32) / np.sqrt(512),
                                       [0, 0, 10, 10])
    database.assegna_volto_a_persona(id_volto, id_p)
    per_persona = database.ottieni_embedding_volti_per_persona()
    assert id_p in per_persona and len(per_persona[id_p]) == 1
    database.rinomina_persona(id_p, "Mario")
    persone = database.ottieni_persone()
    assert persone[0]["name"] == "Mario" and persone[0]["n_volti"] == 1
    media_p = database.ottieni_media_di_persona(id_p)
    assert len(media_p) == 1 and media_p[0]["id"] == id_media
    print("test_database_v07: OK")

if __name__ == "__main__":
    esegui_test()

"""Clustering dei volti: assegnazione greedy e re-cluster DBSCAN."""
import os, tempfile
import numpy as np

import config
config.PERCORSO_DB = os.path.join(tempfile.mkdtemp(), "test.db")
import database
database._store_frame = None  # non toccare l'indice Chroma reale (vedi test_database_v07)
database._store_volti = None
import persone

def _vettore(seme):
    rng = np.random.default_rng(seme)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)

def _variante(base, rumore=0.02, seme=0):
    rng = np.random.default_rng(seme)
    v = base + rumore * rng.normal(size=512).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)

def esegui_test():
    database.inizializza_db()
    id_media = database.aggiungi_elemento_multimediale("/tmp/y.jpg", "y.jpg", "image", 1, "hash_p")
    # carica_tutti_embedding_volti filtra su processed=1: il media va marcato elaborato
    database.aggiorna_stato_elaborazione(id_media=id_media, stato_elaborazione=1)
    id_frame = database.crea_frame(id_media, 0, 0.0, "/tmp/y.jpg")

    alice = _vettore(1)
    bruno = _vettore(2)

    # due volti quasi identici -> stessa persona; uno diverso -> nuova persona
    v1 = database.aggiungi_volto(id_media, id_frame, "/tmp/a1.jpg", alice, [0,0,1,1])
    p1 = persone.assegna_persona(alice, v1)
    v2 = database.aggiungi_volto(id_media, id_frame, "/tmp/a2.jpg", _variante(alice, seme=3), [0,0,1,1])
    p2 = persone.assegna_persona(_variante(alice, seme=3), v2)
    v3 = database.aggiungi_volto(id_media, id_frame, "/tmp/b1.jpg", bruno, [0,0,1,1])
    p3 = persone.assegna_persona(bruno, v3)
    assert p1 == p2, "varianti della stessa faccia devono unirsi"
    assert p3 != p1, "facce diverse devono separarsi"

    # re-cluster completo: stesso risultato (2 persone), nome preservato
    database.rinomina_persona(p1, "Alice")
    n = persone.ricalcola_tutti_cluster()
    assert n == 2, n
    nomi = {p["name"] for p in database.ottieni_persone()}
    assert "Alice" in nomi
    print("test_persone: OK")

if __name__ == "__main__":
    esegui_test()

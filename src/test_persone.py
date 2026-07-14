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

    # una persona "Alice" con volti divisi su due basi lontane (dopo il
    # re-cluster finiscono su due cluster diversi): il nome deve andare
    # a un solo cluster, non a entrambi
    carlo = _vettore(10)
    v4 = database.aggiungi_volto(id_media, id_frame, "/tmp/c1.jpg", carlo, [0,0,1,1])
    id_persona_carlo = database.crea_persona()
    database.assegna_volto_a_persona(v4, id_persona_carlo)
    dora = _vettore(20)
    v5 = database.aggiungi_volto(id_media, id_frame, "/tmp/d1.jpg", dora, [0,0,1,1])
    database.assegna_volto_a_persona(v5, id_persona_carlo)  # stessa vecchia persona, basi lontane
    database.rinomina_persona(id_persona_carlo, "Alice")

    n = persone.ricalcola_tutti_cluster()
    persone_alice = [p for p in database.ottieni_persone() if p["name"] == "Alice"]
    assert len(persone_alice) <= 1, persone_alice
    assert n == 4, n  # p1/p2, p3, carlo, dora: 4 cluster distinti

    # crea_persona_con_volto: persona creata e volto assegnato in un'unica chiamata
    # (atomicita' che evita la potatura di ottieni_persone tra crea e assegna)
    v6 = database.aggiungi_volto(id_media, id_frame, "/tmp/e1.jpg", _vettore(99), [0, 0, 1, 1])
    id_persona_nuova = database.crea_persona_con_volto(v6)
    persone_dopo = {p["id"]: p for p in database.ottieni_persone()}
    assert id_persona_nuova in persone_dopo and persone_dopo[id_persona_nuova]["n_volti"] == 1
    print("test_persone: OK")

if __name__ == "__main__":
    esegui_test()

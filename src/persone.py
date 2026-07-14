"""Raggruppamento dei volti in persone (Face Database).

Strategia incrementale: un volto nuovo entra nella persona con il centroide piu'
simile se il coseno supera config.SOGLIA_SIMILARITA_VOLTI, altrimenti fonda una
persona nuova. Il re-cluster completo (DBSCAN) corregge la deriva del greedy.
"""
# ponytail: assegnazione greedy sul centroide; se le fusioni sbagliate diventano
# un problema, passare a re-cluster automatico periodico (HAC/DBSCAN).
import numpy as np

import config
import database


def _centroidi():
    """{id_persona: centroide normalizzato} dai volti gia' assegnati."""
    risultato = {}
    for id_persona, embedding in database.ottieni_embedding_volti_per_persona().items():
        c = np.mean(np.stack(embedding), axis=0)
        norma = np.linalg.norm(c)
        if norma > 0:
            risultato[id_persona] = c / norma
    return risultato


def assegna_persona(embedding, id_volto):
    """Assegna il volto alla persona piu' vicina (o ne crea una). Ritorna id_persona."""
    migliore_id, migliore_sim = None, -1.0
    for id_persona, centroide in _centroidi().items():
        sim = float(np.dot(embedding, centroide))
        if sim > migliore_sim:
            migliore_id, migliore_sim = id_persona, sim
    if migliore_id is None or migliore_sim < config.SOGLIA_SIMILARITA_VOLTI:
        migliore_id = database.crea_persona()
    database.assegna_volto_a_persona(id_volto, migliore_id)
    return migliore_id


def ricalcola_tutti_cluster():
    """Re-clustering completo con DBSCAN (metrica coseno). Preserva i nomi
    esistenti assegnando ogni nome al SINGOLO cluster dove ha piu' voti (cosi'
    un nome non finisce mai su due cluster diversi se i volti della vecchia
    persona si sono spezzati). Ritorna il numero di persone risultanti."""
    from sklearn.cluster import DBSCAN

    volti = database.carica_tutti_embedding_volti()
    if not volti:
        database.azzera_persone()
        return 0

    # nome della vecchia persona per ogni volto (per preservarlo dopo)
    vecchi_nomi = {p["id"]: p["name"] for p in database.ottieni_persone() if p["name"]}
    connessione = database.ottieni_connessione()
    volto_a_vecchia_persona = dict(connessione.execute(
        "SELECT id, person_id FROM faces WHERE person_id IS NOT NULL").fetchall())
    connessione.close()

    matrice = np.stack([v["embedding"] for v in volti])
    # eps in distanza coseno = 1 - soglia di similarita'
    etichette = DBSCAN(eps=1.0 - config.SOGLIA_SIMILARITA_VOLTI, min_samples=1,
                       metric="cosine").fit_predict(matrice)

    database.azzera_persone()
    persona_per_etichetta = {}
    voti = {}  # (nome, etichetta) -> conteggio
    for volto, etichetta in zip(volti, etichette):
        if etichetta not in persona_per_etichetta:
            persona_per_etichetta[etichetta] = database.crea_persona()
        database.assegna_volto_a_persona(volto["face_id"], persona_per_etichetta[etichetta])
        vecchio_nome = vecchi_nomi.get(volto_a_vecchia_persona.get(volto["face_id"]))
        if vecchio_nome:
            chiave = (vecchio_nome, etichetta)
            voti[chiave] = voti.get(chiave, 0) + 1

    # per ogni nome, il cluster dove ha piu' voti (un solo cluster per nome)
    cluster_per_nome = {}
    for (nome, etichetta), conteggio in voti.items():
        migliore = cluster_per_nome.get(nome)
        if migliore is None or conteggio > migliore[1]:
            cluster_per_nome[nome] = (etichetta, conteggio)

    # assegno i nomi ai cluster in ordine di voti decrescenti: se due nomi
    # vogliono lo stesso cluster vince quello con piu' voti, l'altro si perde
    # (un cluster -> al massimo un nome)
    etichette_assegnate = set()
    for nome, (etichetta, conteggio) in sorted(cluster_per_nome.items(), key=lambda x: -x[1][1]):
        if etichetta in etichette_assegnate:
            continue  # cluster gia' assegnato a un nome con piu' voti: questo nome si perde
        database.rinomina_persona(persona_per_etichetta[etichetta], nome)
        etichette_assegnate.add(etichetta)

    return len(persona_per_etichetta)

import os
import numpy as np
import chromadb
from typing import List, Dict, Any, Tuple


# Un solo PersistentClient per cartella, riusato se la stessa cartella viene richiesta più volte.
# In ChromaDB 1.5.x due collezioni HNSW che vivono nello STESSO path/processo si
# interferiscono: dopo aver interrogato "faces", la query su "qwen_frames" fallisce con
# "Error creating hnsw segment reader: Nothing found on disk" (accade anche con count()/get()).
# Per questo ogni collezione usa una SOTTOCARTELLA separata (vedi ChromaStore.__init__),
# così ognuna ha il proprio client isolato e il baco non si manifesta.
_cache_client: Dict[str, Any] = {}


def _ottieni_client(percorso_db: str):
    client = _cache_client.get(percorso_db)
    if client is None:
        client = chromadb.PersistentClient(path=percorso_db)
        _cache_client[percorso_db] = client
    return client


class ChromaStore:
    """Implementazione concreta dell'archivio vettoriale utilizzando ChromaDB.

    La collezione viene salvata in una cartella locale chiamata `chroma_db`
    nella cartella radice del progetto. Ogni vettore è memorizzato insieme
    ai metadati forniti dall'utente.
    """

    def __init__(self, nome_collezione: str = "media_vectors"):
        # Ogni collezione vive in una SOTTOCARTELLA dedicata di chroma_db (es.
        # chroma_db/qwen_frames, chroma_db/faces) così ognuna ha un client/indice
        # isolato: aggira il baco di interferenza HNSW tra collezioni di ChromaDB 1.5.x
        # (vedi nota su _cache_client). Questo file vive in .\src\, quindi si risale di
        # un livello per tenere chroma_db nella cartella radice del progetto.
        percorso_db = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db", nome_collezione)
        os.makedirs(percorso_db, exist_ok=True)
        self.client = _ottieni_client(percorso_db)

        # Recupera o crea la collezione vettoriale
        self.collezione = self.client.get_or_create_collection(name=nome_collezione)
        self._nome = nome_collezione

    def aggiungi_o_aggiorna(self, id_elementi: List[int], vettori: List[np.ndarray],
                             metadati: List[Dict[str, Any]]) -> None:
        """Inserisce o aggiorna un gruppo di vettori con i relativi metadati.

        * `id_elementi` – lista di identificatori numerici (convertiti in stringhe per Chroma).
        * `vettori` – lista di array NumPy (float32). Vengono convertiti in
          liste standard Python poiché Chroma richiede dati serializzabili in JSON.
          * `metadati` – dizionari con i metadati associati a ciascun vettore.
        """
        id_stringhe = [str(i) for i in id_elementi]
        vettori_lista = [v.tolist() for v in vettori]
        self.collezione.upsert(ids=id_stringhe, embeddings=vettori_lista, metadatas=metadati)

    def cerca_simili(self, vettore: np.ndarray, migliori_k: int = 10,
                     filtro: Dict[str, Any] | None = None) -> List[Tuple[int, float, Dict]]:
        """Restituisce i `migliori_k` vettori più simili.

        Di default Chroma restituisce le distanze L2. Le convertiamo in un punteggio
        di similarità compreso tra 0 e 1 tramite la formula `1 - distanza`. Questo metodo
        funziona bene per gli embedding normalizzati di CLIP.
        
        Restituisce una lista di tuple: (id_documento, punteggio_similarita, metadati).
        """
        risultati = self.collezione.query(
            query_embeddings=[vettore.tolist()],
            n_results=migliori_k,
            where=filtro,
        )
        id_stringhe = risultati["ids"][0]
        distanze = risultati["distances"][0]
        metadati_risultato = risultati["metadatas"][0]
        
        # Converte le distanze L2 in punteggi di similarità
        punteggi_similarita = [1.0 - d for d in distanze]
        return [(int(i), sim, meta) for i, sim, meta in zip(id_stringhe, punteggi_similarita, metadati_risultato)]

    def elimina(self, id_elementi: List[int]) -> None:
        """Rimuove dall'archivio vettoriale i vettori identificati dagli `id_elementi`."""
        if not id_elementi:
            return
        self.collezione.delete(ids=[str(i) for i in id_elementi])

    def conteggio(self) -> int:
        """Restituisce il numero di vettori presenti nella collezione."""
        return self.collezione.count()

    def ottieni_nome_collezione(self) -> str:
        """Restituisce il nome della collezione vettoriale sottostante."""
        return self._nome

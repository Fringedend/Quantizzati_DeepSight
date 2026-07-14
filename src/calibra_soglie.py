"""Misura le distribuzioni dei coseni Qwen sull'archivio reale, per tarare:
FATTORE_SIGMA_RICERCA_TESTO, SOGLIA_SIMILARITA_IMMAGINE, SCALA_LOGIT_TAG.
Uso: python calibra_soglie.py "query di prova" [altra query ...]
"""
import sys
import numpy as np
import database
from models import gestore

def esegui():
    frames = database.carica_tutti_embedding_clip()
    if len(frames) < 5:
        print("Archivio troppo piccolo: importa ed elabora almeno 5 elementi.")
        return
    matrice = np.stack([f["embedding"] for f in frames])
    print(f"{len(frames)} frame con embedding.")

    # immagine->immagine: distribuzione dei coseni tra tutte le coppie
    coseni_img = (matrice @ matrice.T)[np.triu_indices(len(frames), k=1)]
    print(f"img->img   media={coseni_img.mean():.3f} sigma={coseni_img.std():.3f} "
          f"p95={np.percentile(coseni_img, 95):.3f}  (SOGLIA_SIMILARITA_IMMAGINE ~ p95)")

    qwen = gestore.ottieni_qwen()
    for query in sys.argv[1:] or ["una spiaggia al tramonto", "un documento con del testo"]:
        v = qwen.ottieni_embedding_testo(query)
        coseni = matrice @ v
        print(f"testo->img '{query}': media={coseni.mean():.3f} sigma={coseni.std():.3f} "
              f"max={coseni.max():.3f}")

if __name__ == "__main__":
    esegui()

"""Check dell'assunzione su cui si regge cerca_frame_simili/cerca_volti_simili:
per vettori normalizzati l'ordinamento per distanza L2 (quello che usa Chroma di
default) coincide con l'ordinamento per coseno decrescente. Quindi Chroma ci dà
gli stessi candidati top-k del coseno e poi ricalcoliamo il coseno esatto.
"""
import numpy as np


def _norm(v):
    return v / np.linalg.norm(v)


def test_l2_rank_uguale_coseno():
    rng = np.random.default_rng(0)
    q = _norm(rng.standard_normal(512))
    vs = [_norm(rng.standard_normal(512)) for _ in range(200)]

    per_coseno = sorted(range(len(vs)), key=lambda i: -float(np.dot(q, vs[i])))
    per_l2 = sorted(range(len(vs)), key=lambda i: float(np.sum((q - vs[i]) ** 2)))
    assert per_coseno == per_l2


if __name__ == "__main__":
    test_l2_rank_uguale_coseno()
    print("OK: ranking L2 == ranking coseno per vettori normalizzati")

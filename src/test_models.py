"""Diagnostica: verifica che i modelli AI (Qwen, FaceNet, Whisper) si carichino."""
import numpy as np
from models import gestore
import config

def esegui_test():
    print(f"Dispositivo: {gestore.dispositivo}")

    print("1/3 Qwen3-VL-Embedding (llama-server)...")
    qwen = gestore.ottieni_qwen()
    v_testo = qwen.ottieni_embedding_testo("una prova")
    assert v_testo.shape == (config.DIM_EMBEDDING_QWEN,), v_testo.shape
    assert abs(float(np.linalg.norm(v_testo)) - 1.0) < 1e-3
    print("   OK: embedding testo 2048-d normalizzato")

    print("2/3 MTCNN + FaceNet...")
    gestore.ottieni_volti()
    print("   OK")

    print("3/3 Whisper...")
    gestore.ottieni_whisper()
    print("   OK")
    print("Tutti i modelli caricati correttamente.")

if __name__ == "__main__":
    esegui_test()

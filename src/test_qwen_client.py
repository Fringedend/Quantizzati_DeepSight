"""Verifica smart_resize e la conversione base64 (senza server)."""
from PIL import Image
import qwen_client

def esegui_test():
    # multipli di 32, aspect ratio preservata
    h, w = qwen_client.smart_resize(1080, 1920)
    assert h % 32 == 0 and w % 32 == 0, (h, w)
    assert h * w <= 1843200
    # immagine minuscola: portata almeno a min_pixels
    h2, w2 = qwen_client.smart_resize(10, 10)
    assert h2 % 32 == 0 and w2 % 32 == 0 and h2 * w2 >= 4096
    # base64 di una PIL RGB
    img = Image.new("RGB", (100, 60), "red")
    b64 = qwen_client.pil_to_base64_qwen(img)
    assert isinstance(b64, str) and len(b64) > 100
    print("test_qwen_client: OK")

if __name__ == "__main__":
    esegui_test()

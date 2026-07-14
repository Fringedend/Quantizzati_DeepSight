import os
import sys
import torch

# Con stdout ridirezionato (pipe/file) Windows usa cp1252: la progress bar di
# download di EasyOCR stampa '█' (U+2588) e manderebbe in crash il caricamento.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Assicura che l'eseguibile di imageio-ffmpeg sia nel PATH in modo che Whisper possa trovare FFmpeg
try:
    import imageio_ffmpeg
    percorso_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cartella_ffmpeg = os.path.dirname(percorso_ffmpeg)
    os.environ["PATH"] = cartella_ffmpeg + os.pathsep + os.environ.get("PATH", "")
except Exception as errore:
    print("Avviso: Impossibile impostare la cartella FFmpeg nel PATH:", errore)

print("Versione Python:", sys.version)
print("Versione PyTorch:", torch.__version__)
print("CUDA disponibile (GPU):", torch.cuda.is_available())
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo in uso: {dispositivo}")

# 1. Test di CLIP
print("\n--- Test di CLIP ---")
try:
    from transformers import CLIPProcessor, CLIPModel
    print("Caricamento del modello CLIP...")
    modello = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processore = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP caricato con successo!")
except Exception as errore:
    print("Caricamento di CLIP fallito:", errore)

# 2. Test di FaceNet / MTCNN
print("\n--- Test di MTCNN & FaceNet ---")
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    print("Caricamento di MTCNN & FaceNet...")
    mtcnn = MTCNN(keep_all=True, device="cpu") # Forza l'uso della CPU per il test
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("Modelli facciali caricati con successo!")
except Exception as errore:
    print("Caricamento dei modelli facciali fallito:", errore)

# 3. Test di EasyOCR
print("\n--- Test di EasyOCR ---")
try:
    import easyocr
    print("Caricamento di EasyOCR...")
    lettore = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR caricato con successo!")
except Exception as errore:
    print("Caricamento di EasyOCR fallito:", errore)

# 4. Test di Whisper
print("\n--- Test di Whisper ---")
try:
    import whisper
    print("Caricamento di Whisper...")
    modello_whisper = whisper.load_model("tiny", device="cpu")
    print("Whisper caricato con successo!")
except Exception as errore:
    print("Caricamento di Whisper fallito:", errore)

print("\nTest completati!")

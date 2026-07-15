import os

# Cartella radice del progetto
# (questo file vive in .\src\, quindi si risale di un livello per tenere i
# dati nella cartella principale del progetto.)
DIR_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cartelle dei dati (relative a DIR_BASE)
DIR_DATI = os.path.join(DIR_BASE, "data")
DIR_ARCHIVIO = os.path.join(DIR_DATI, "archive")
DIR_FRAME = os.path.join(DIR_DATI, "frames")
DIR_VOLTI = os.path.join(DIR_DATI, "faces")
DIR_ANTEPRIME = os.path.join(DIR_DATI, "thumbnails")
DIR_DB = os.path.join(DIR_DATI, "db")
DIR_QUARANTENA = os.path.join(DIR_DATI, "quarantena")  # file intrusi isolati (non indicizzati)
PERCORSO_DB = os.path.join(DIR_DB, "archive.db")

# Assicura che tutte le cartelle necessarie esistano
for cartella in [DIR_DATI, DIR_ARCHIVIO, DIR_FRAME, DIR_VOLTI, DIR_ANTEPRIME, DIR_DB, DIR_QUARANTENA]:
    os.makedirs(cartella, exist_ok=True)

# Configurazione dei modelli di Intelligenza Artificiale
NOME_MODELLO_WHISPER = "base"  # opzioni disponibili: "tiny", "base", "small", "medium"

# --- Qwen3-VL-Embedding-2B (embedding multimodali immagine+testo, via llama-server) ---
# Sostituisce CLIP (ricerca semantica, similarità, tag) ed EasyOCR (il modello "legge"
# il testo nelle immagini a livello di embedding). Gira come sottoprocesso llama-server
# su CPU, gestito pigramente da models.GestoreQwen.
DIR_MODELLI = os.path.join(DIR_BASE, "models", "qwen")
_NOME_SERVER = "llama-server.exe" if os.name == "nt" else "llama-server"
PERCORSO_LLAMA_SERVER = os.path.join(DIR_MODELLI, _NOME_SERVER)
PERCORSO_MODELLO_QWEN = os.path.join(DIR_MODELLI, "Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf")
PERCORSO_MMPROJ_QWEN = os.path.join(DIR_MODELLI, "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf")
QWEN_HOST = "127.0.0.1"
QWEN_PORT = 8091
# Un core resta libero per la UI: llama-server a piena CPU congela gli
# aggiornamenti del pannello coda di Streamlit durante gli embedding.
QWEN_THREADS = max(1, (os.cpu_count() or 2) - 1)
# Strati Qwen offloadati sulla GPU: 99 = tutti, 0 = forza CPU. Applicato solo se il
# dispositivo rilevato e' 'cuda' (vedi models.GestoreModelli.ottieni_qwen); la build CPU
# di llama-server lo ignora comunque, e se l'avvio GPU fallisce si ripiega su CPU.
QWEN_NGL = 99
DIM_EMBEDDING_QWEN = 2048  # 2B = 2048-d; NON mescolare con modelli di dimensione diversa
# Istruzione di retrieval per le QUERY testuali (i documenti/immagini NON ricevono istruzione)
ISTRUZIONE_RICERCA = "Retrieve images or text relevant to the user's query."

# Prompt negativo: punteggio finale = cos(query, img) - LAMBDA * cos(negativo, img)
LAMBDA_PROMPT_NEGATIVO = 0.5

# Tag zero-shot con Qwen: softmax(similarita' * SCALA_LOGIT_TAG). Qwen non ha la
# logit_scale appresa di CLIP: la scala e' una costante da tarare (vedi calibra_soglie.py).
SCALA_LOGIT_TAG = 40.0

# Configurazione del dispositivo hardware: 'auto', 'cpu', 'cuda'
MODALITA_DISPOSITIVO = "auto"

# Parametri per l'elaborazione dei video
# Numero di secondi di intervallo tra l'estrazione di un frame e il successivo
INTERVALLO_FRAME_VIDEO = 3.0  # Estrae 1 frame ogni 3 secondi

# Parametri per il riconoscimento facciale
# Soglia coseno per considerare due volti la stessa persona. Con gli embedding FaceNet
# (vggface2) persone diverse stanno tipicamente sotto 0.5 e la stessa persona sopra ~0.7;
# a 0.60 illustrazioni e volti stilizzati producevano falsi positivi (es. un ritratto reale
# che "corrispondeva" a un disegno al 68%), perciò è alzata a 0.72.
SOGLIA_SIMILARITA_VOLTI = 0.72
DIMENSIONE_MINIMA_VOLTO = 40  # Dimensione minima del volto in pixel (volti troppo piccoli => embedding inaffidabili)
# Confidenza minima di MTCNN per accettare un rilevamento: filtra i falsi positivi
# (volti "visti" in trame e disegni, es. il cestello di un'asciugatrice ~0.83), lasciando
# passare i volti reali (tipicamente >=0.93). È il filtro discriminante principale, più
# della dimensione: alzarla oltre ~0.92 inizia a scartare anche ritratti veri.
SOGLIA_CONFIDENZA_RILEVAMENTO_VOLTO = 0.90

# Nota (v0.7): le ricerche Qwen (testo->immagine e immagine->immagine) NON hanno soglia:
# i risultati sono sempre ordinati per rilevanza e la UI mostra i migliori 5 + "Mostra altri".
# Le vecchie soglie CLIP (adattiva testo, SOGLIA_SIMILARITA_IMMAGINE) sono state rimosse.

# Impostazioni per la classificazione dei tag (basata sulla probabilità softmax)
# Probabilità softmax minima affinché una categoria venga assegnata come tag.
SOGLIA_PROBABILITA_TAG = 0.05  # Soglia minima del 5%
MASSIMO_TAG_PER_IMMAGINE = 10  # Numero massimo di tag assegnati per immagine/frame

def ottieni_archivio_vettoriale(nome_collezione="media_vectors"):
    """Restituisce un'istanza di ChromaStore per la collezione vettoriale indicata."""
    from chroma_store import ChromaStore
    return ChromaStore(nome_collezione=nome_collezione)

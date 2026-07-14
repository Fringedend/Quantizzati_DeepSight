import os
import threading
import torch
import numpy as np
import torchvision.transforms as T
import config

# Configura il percorso di FFmpeg nell'ambiente in modo che Whisper possa trovarlo
try:
    import imageio_ffmpeg
    cartella_ffmpeg = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = cartella_ffmpeg + os.pathsep + os.environ.get("PATH", "")
except Exception as errore:
    print(f"Attenzione: Impossibile impostare PATH per FFmpeg tramite imageio-ffmpeg: {errore}")

class GestoreCLIP:
    """Classe di supporto per il modello CLIP (generazione embedding immagine e testo)."""
    
    def __init__(self, dispositivo):
        self.dispositivo = dispositivo
        from transformers import CLIPProcessor, CLIPModel
        print(f"Caricamento del modello CLIP '{config.NOME_MODELLO_CLIP}' su {dispositivo}...")
        self.modello = CLIPModel.from_pretrained(config.NOME_MODELLO_CLIP).to(dispositivo)
        self.processore = CLIPProcessor.from_pretrained(config.NOME_MODELLO_CLIP)
        self.modello.eval()
        # Encoder di testo multilingue caricato pigramente (solo alla prima query di ricerca).
        self._modello_testo_multilingue = None
        print("Modello CLIP caricato con successo.")

    def ottieni_embedding_query_testo(self, testo):
        """Embedding normalizzato a 512-d di una query di ricerca, con encoder multilingue.

        A differenza di ottieni_embedding_testo (encoder inglese di CLIP, usato per il
        tagging), questo mappa testo in molte lingue — italiano incluso — nello stesso
        spazio dell'encoder immagini, così la ricerca semantica in italiano è precisa.
        """
        if self._modello_testo_multilingue is None:
            from sentence_transformers import SentenceTransformer
            nome = config.NOME_MODELLO_CLIP_TESTO_MULTILINGUE
            print(f"Caricamento dell'encoder di testo multilingue '{nome}' su {self.dispositivo}...")
            self._modello_testo_multilingue = SentenceTransformer(nome, device=self.dispositivo)
            print("Encoder di testo multilingue caricato con successo.")
        # normalize_embeddings=True → norma unitaria, così np.dot restituisce il coseno
        return self._modello_testo_multilingue.encode(
            testo, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def ottieni_embedding_immagine(self, immagine_pil):
        """Genera un embedding normalizzato a 512 dimensioni per un'immagine PIL."""
        ingressi = self.processore(images=immagine_pil, return_tensors="pt").to(self.dispositivo)
        with torch.no_grad():
            caratteristiche_immagine = self.modello.get_image_features(**ingressi)
        
        # Gestisce BaseModelOutputWithPooling restituito dalle versioni più recenti delle librerie transformers
        if hasattr(caratteristiche_immagine, "pooler_output") and caratteristiche_immagine.pooler_output is not None:
            caratteristiche_immagine = caratteristiche_immagine.pooler_output
            
        # Normalizza il vettore a lunghezza unitaria per la similarità del coseno
        caratteristiche_immagine = caratteristiche_immagine / caratteristiche_immagine.norm(dim=-1, keepdim=True)
        return caratteristiche_immagine.cpu().numpy()[0]

    def ottieni_embedding_testo(self, testo):
        """Genera un embedding normalizzato a 512 dimensioni per una query di testo."""
        ingressi = self.processore(text=[testo], return_tensors="pt", padding=True).to(self.dispositivo)
        with torch.no_grad():
            caratteristiche_testo = self.modello.get_text_features(**ingressi)
            
        # Gestisce BaseModelOutputWithPooling
        if hasattr(caratteristiche_testo, "pooler_output") and caratteristiche_testo.pooler_output is not None:
            caratteristiche_testo = caratteristiche_testo.pooler_output
            
        # Normalizza a lunghezza unitaria
        caratteristiche_testo = caratteristiche_testo / caratteristiche_testo.norm(dim=-1, keepdim=True)
        return caratteristiche_testo.cpu().numpy()[0]

    def ottieni_scala_logit(self):
        """Restituisce la scala logit (temperatura) appresa da CLIP per il calcolo del softmax."""
        with torch.no_grad():
            return self.modello.logit_scale.exp().item()

class GestoreQwen:
    """Embedding multimodali Qwen3-VL-2B tramite sottoprocesso llama-server (CPU).

    Il server viene avviato pigramente al primo embedding e fermato da
    libera_memoria() o alla chiusura del processo (atexit).
    """

    def __init__(self):
        import qwen_client
        self._qc = qwen_client
        self.server = qwen_client.LlamaServer(
            exe=config.PERCORSO_LLAMA_SERVER,
            model_path=config.PERCORSO_MODELLO_QWEN,
            mmproj_path=config.PERCORSO_MMPROJ_QWEN,
            host=config.QWEN_HOST,
            port=config.QWEN_PORT,
            threads=config.QWEN_THREADS,
        )
        for percorso in (config.PERCORSO_LLAMA_SERVER, config.PERCORSO_MODELLO_QWEN,
                         config.PERCORSO_MMPROJ_QWEN):
            if not os.path.exists(percorso):
                raise FileNotFoundError(
                    f"File del modello Qwen mancante: {percorso}. "
                    "Copia i file in models/qwen/ (vedi README).")
        print("Avvio llama-server (Qwen3-VL-Embedding-2B)...")
        self.server.avvia()
        import atexit
        atexit.register(self.server.ferma)
        print("llama-server pronto.")

    def ottieni_embedding_testo(self, testo, con_istruzione=True):
        """Embedding 2048-d normalizzato di un testo. Le QUERY di ricerca usano
        l'istruzione di retrieval; i testi 'documento' (es. nomi di categorie) no."""
        istruzione = config.ISTRUZIONE_RICERCA if con_istruzione else None
        return self.server.embed_text(testo, instruction=istruzione)

    def ottieni_embedding_immagine(self, sorgente):
        """Embedding 2048-d normalizzato di un'immagine (percorso file o PIL)."""
        from PIL import Image
        img = Image.open(sorgente) if isinstance(sorgente, str) else sorgente
        return self.server.embed_image_b64(self._qc.pil_to_base64_qwen(img))

class RiconoscitoreVolti:
    """Classe per rilevare volti e generare i relativi embedding con MTCNN e FaceNet."""
    
    def __init__(self, dispositivo):
        self.dispositivo = dispositivo
        from facenet_pytorch import MTCNN, InceptionResnetV1
        print(f"Caricamento di MTCNN e FaceNet (InceptionResnetV1) su {dispositivo}...")
        
        self.mtcnn = MTCNN(
            keep_all=True, 
            min_face_size=config.DIMENSIONE_MINIMA_VOLTO, 
            device=dispositivo,
            post_process=False
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(dispositivo)
        
        # Pipeline di normalizzazione dell'immagine richiesta da FaceNet
        self.trasformazione = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Mappa i valori nell'intervallo [-1, 1]
        ])
        print("Modelli per i volti caricati con successo.")

    def rileva_e_codifica_volti(self, immagine_pil):
        """
        Rileva tutti i volti presenti in un'immagine PIL.
        
        Restituisce una lista di dizionari:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "crop": PIL.Image (ritaglio del volto),
                "embedding": np.ndarray (embedding da 512 dimensioni)
            },
            ...
        ]
        """
        larghezza, altezza = immagine_pil.size
        # Rileva i volti: MTCNN restituisce le coordinate dei rettangoli e le probabilità associate
        riquadri, probabilita = self.mtcnn.detect(immagine_pil)
        if riquadri is None or len(riquadri) == 0:
            return []
            
        risultati = []
        for riquadro, prob in zip(riquadri, probabilita):
            if prob is None or prob < config.SOGLIA_CONFIDENZA_RILEVAMENTO_VOLTO: # Ignora i rilevamenti a bassa confidenza
                continue
                
            # Limita le coordinate entro i bordi dell'immagine originale
            x1 = max(0, int(riquadro[0]))
            y1 = max(0, int(riquadro[1]))
            x2 = min(larghezza, int(riquadro[2]))
            y2 = min(altezza, int(riquadro[3]))
            
            if (x2 - x1) < config.DIMENSIONE_MINIMA_VOLTO or (y2 - y1) < config.DIMENSIONE_MINIMA_VOLTO:
                continue
                
            # Esegue il ritaglio del volto dall'immagine originale
            ritaglio = immagine_pil.crop((x1, y1, x2, y2))
            
            # Pre-elaborazione del ritaglio per FaceNet
            try:
                tensore_volto = self.trasformazione(ritaglio).unsqueeze(0).to(self.dispositivo)
                with torch.no_grad():
                    embedding = self.resnet(tensore_volto).cpu().numpy()[0]
                
                # Normalizza l'embedding per l'utilizzo della similarità del coseno
                norma = np.linalg.norm(embedding)
                if norma > 0:
                    embedding = embedding / norma
                    
                risultati.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(prob),
                    "crop": ritaglio,
                    "embedding": embedding
                })
            except Exception as errore:
                print(f"Generazione dell'embedding fallita per il volto {riquadro}: {errore}")
                
        return risultati

class GestoreOCR:
    """Classe di supporto per l'estrazione di testo da immagini tramite EasyOCR."""
    
    def __init__(self, dispositivo):
        import easyocr
        usa_gpu = (dispositivo == "cuda")
        print(f"Caricamento di EasyOCR (gpu={usa_gpu}) per le lingue {config.LINGUE_EASYOCR}...")
        # verbose=False: la progress bar di download di EasyOCR stampa '█' (U+2588),
        # che crasha se stdout è ridirezionato su Windows (encoding cp1252).
        self.lettore = easyocr.Reader(config.LINGUE_EASYOCR, gpu=usa_gpu, verbose=False)
        print("EasyOCR caricato con successo.")

    def estrai_testo(self, img_np_o_percorso):
        """
        Estrae tutto il testo rilevato da un'immagine.
        Accetta sia un percorso file sia un array NumPy (formato OpenCV BGR/RGB).
        Restituisce una stringa di testo concatenata.
        """
        try:
            # readtext restituisce: [([[x,y],[x,y],[x,y],[x,y]], testo, confidenza), ...]
            risultati = self.lettore.readtext(img_np_o_percorso)
            testi = [r[1] for r in risultati if r[2] > 0.35] # Filtra letture a bassa confidenza
            return " ".join(testi).strip()
        except Exception as errore:
            print(f"Lettura OCR fallita: {errore}")
            return ""

class GestoreWhisper:
    """Classe di supporto per la trascrizione audio tramite Whisper."""
    
    def __init__(self, dispositivo):
        self.dispositivo = dispositivo
        import whisper
        print(f"Caricamento del modello Whisper '{config.NOME_MODELLO_WHISPER}' su {dispositivo}...")
        self.modello = whisper.load_model(config.NOME_MODELLO_WHISPER, device=dispositivo)
        print("Whisper caricato con successo.")

    def trascrivi(self, percorso_audio):
        """Effettua la trascrizione dell'audio da un file WAV. Restituisce il testo."""
        if not os.path.exists(percorso_audio):
            return ""
        try:
            # Se usiamo la CPU disabilitiamo fp16 per evitare avvisi da PyTorch
            valore_fp16 = (self.dispositivo == "cuda")
            # Carichiamo il WAV manualmente e passiamo l'array a Whisper: in questo modo
            # Whisper non invoca internamente il comando 'ffmpeg' dal PATH (che su Windows
            # non è disponibile, dato che imageio-ffmpeg fornisce un eseguibile con un nome
            # diverso da 'ffmpeg.exe'), evitando l'errore [WinError 2].
            audio = self._carica_wav(percorso_audio)
            risultato = self.modello.transcribe(audio, fp16=valore_fp16)
            return risultato.get("text", "").strip()
        except Exception as errore:
            print(f"Trascrizione Whisper fallita: {errore}")
            return ""

    @staticmethod
    def _carica_wav(percorso_audio):
        """Legge un WAV PCM 16-bit (mono, 16kHz) in un array float32 normalizzato per Whisper."""
        import wave
        with wave.open(percorso_audio, "rb") as file_wav:
            n_canali = file_wav.getnchannels()
            frame = file_wav.readframes(file_wav.getnframes())
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if n_canali > 1:
            # Converte in mono facendo la media dei canali
            audio = audio.reshape(-1, n_canali).mean(axis=1)
        return audio

class GestoreModelli:
    """Gestisce il caricamento pigro (lazy-loading) e la cache dei modelli AI locali."""
    
    def __init__(self):
        # Determina il dispositivo da utilizzare per l'esecuzione
        gpu_disponibile = torch.cuda.is_available()
        if config.MODALITA_DISPOSITIVO == "cuda":
            # Richiesta esplicita di GPU: se non è disponibile ricadiamo su CPU
            # con un avviso, invece di far crashare l'applicazione.
            if gpu_disponibile:
                self.dispositivo = "cuda"
            else:
                print("Attenzione: MODALITA_DISPOSITIVO='cuda' ma nessuna GPU CUDA "
                      "è disponibile. Utilizzo della CPU.")
                self.dispositivo = "cpu"
        elif config.MODALITA_DISPOSITIVO == "cpu":
            self.dispositivo = "cpu"
        else: # "auto"
            self.dispositivo = "cuda" if gpu_disponibile else "cpu"
            
        self._clip = None
        self._volti = None
        self._ocr = None
        self._whisper = None
        self._qwen = None
        # Il caricamento pigro può avvenire sia dalla UI sia dal thread di
        # elaborazione in background: il lock evita doppi caricamenti concorrenti.
        self._lock = threading.Lock()

    def ottieni_clip(self) -> GestoreCLIP:
        with self._lock:
            if self._clip is None:
                self._clip = GestoreCLIP(self.dispositivo)
        return self._clip

    def ottieni_volti(self) -> RiconoscitoreVolti:
        with self._lock:
            if self._volti is None:
                self._volti = RiconoscitoreVolti(self.dispositivo)
        return self._volti

    def ottieni_ocr(self) -> GestoreOCR:
        with self._lock:
            if self._ocr is None:
                self._ocr = GestoreOCR(self.dispositivo)
        return self._ocr

    def ottieni_whisper(self) -> GestoreWhisper:
        with self._lock:
            if self._whisper is None:
                self._whisper = GestoreWhisper(self.dispositivo)
        return self._whisper

    def ottieni_qwen(self) -> GestoreQwen:
        with self._lock:
            if self._qwen is None:
                self._qwen = GestoreQwen()
        return self._qwen

    def libera_memoria(self):
        """Rilascia i modelli per liberare la memoria GPU e RAM quando richiesto."""
        self._clip = None
        self._volti = None
        self._ocr = None
        self._whisper = None
        if self._qwen is not None:
            self._qwen.server.ferma()
        self._qwen = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

# Istanza condivisa (Singleton)
gestore = GestoreModelli()

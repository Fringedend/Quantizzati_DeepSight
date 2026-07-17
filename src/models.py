import os
import threading
import config
import torch
import numpy as np
import torchvision.transforms as T

# Configura il percorso di FFmpeg nell'ambiente in modo che Whisper possa trovarlo
try:
    import imageio_ffmpeg
    cartella_ffmpeg = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = cartella_ffmpeg + os.pathsep + os.environ.get("PATH", "")
except Exception as errore:
    print(f"Attenzione: Impossibile impostare PATH per FFmpeg tramite imageio-ffmpeg: {errore}")

class GestoreQwen:
    """Embedding multimodali Qwen3-VL-2B tramite sottoprocesso llama-server (CPU).

    Il server viene avviato pigramente al primo embedding e fermato da
    libera_memoria() o alla chiusura del processo (atexit).
    """

    def __init__(self, n_gpu_layers: int = 0):
        import qwen_client
        self._qc = qwen_client
        self.server = qwen_client.LlamaServer(
            exe=config.PERCORSO_LLAMA_SERVER,
            model_path=config.PERCORSO_MODELLO_QWEN,
            mmproj_path=config.PERCORSO_MMPROJ_QWEN,
            host=config.QWEN_HOST,
            port=config.QWEN_PORT,
            threads=config.QWEN_THREADS,
            n_gpu_layers=n_gpu_layers,
        )
        for percorso in (config.PERCORSO_LLAMA_SERVER, config.PERCORSO_MODELLO_QWEN,
                         config.PERCORSO_MMPROJ_QWEN):
            if not os.path.exists(percorso):
                raise FileNotFoundError(
                    f"File del modello Qwen mancante: {percorso}. "
                    "Copia i file in models/qwen/ (vedi README).")
        print("Avvio llama-server (Qwen3-VL-Embedding-2B)...")
        try:
            self.server.avvia()
        except Exception:
            # Ripiego automatico su CPU: se l'avvio con offload GPU fallisce (VRAM
            # insufficiente o driver CUDA incompatibile con la build) si riprova senza
            # GPU, così su macchine con GPU inadeguata l'embedding funziona comunque.
            if n_gpu_layers > 0:
                print("Avvio Qwen su GPU fallito (VRAM insufficiente o driver CUDA "
                      "incompatibile); ripiego su CPU...")
                self.server.n_gpu_layers = 0
                self.server.avvia()
            else:
                raise
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

class GestoreWhisper:
    """Classe di supporto per la trascrizione audio tramite Whisper."""
    
    def __init__(self, dispositivo):
        self.dispositivo = dispositivo
        import whisper
        print(f"Caricamento del modello Whisper '{config.NOME_MODELLO_WHISPER}' su {dispositivo}...")
        self.modello = whisper.load_model(
            config.NOME_MODELLO_WHISPER,
            device=dispositivo,
            download_root=config.DIR_MODELLI_WHISPER,
        )
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
            
        self._volti = None
        self._whisper = None
        self._qwen = None
        # Il caricamento pigro può avvenire sia dalla UI sia dal thread di
        # elaborazione in background: il lock evita doppi caricamenti concorrenti.
        self._lock = threading.Lock()

    def ottieni_volti(self) -> RiconoscitoreVolti:
        with self._lock:
            if self._volti is None:
                self._volti = RiconoscitoreVolti(self.dispositivo)
        return self._volti

    def ottieni_whisper(self) -> GestoreWhisper:
        with self._lock:
            if self._whisper is None:
                self._whisper = GestoreWhisper(self.dispositivo)
        return self._whisper

    def ottieni_qwen(self) -> GestoreQwen:
        with self._lock:
            if self._qwen is None:
                # Offload GPU solo se il dispositivo rilevato è CUDA; su CPU resta 0.
                # La build CPU di llama-server ignora comunque il flag, e GestoreQwen
                # ripiega su CPU se l'avvio con GPU fallisce.
                ngl = config.QWEN_NGL if self.dispositivo == "cuda" else 0
                self._qwen = GestoreQwen(n_gpu_layers=ngl)
        return self._qwen

    def libera_memoria(self):
        """Rilascia i modelli per liberare la memoria GPU e RAM quando richiesto."""
        self._volti = None
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

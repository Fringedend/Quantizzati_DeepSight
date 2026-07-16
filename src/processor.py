import os
import hashlib
import shutil
import datetime
import subprocess
import tempfile
import threading
import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS

import config
import database
from models import gestore

# Estensioni dei file multimediali gestiti dall'applicazione
ESTENSIONI_IMMAGINI = ['.jpg', '.jpeg', '.png']
# Le GIF (animate) passano dalla pipeline video: OpenCV ne estrae i frame come
# da un normale video. Non hanno audio: la trascrizione viene saltata in _prepara_media.
ESTENSIONI_VIDEO = ['.mp4', '.avi', '.mov', '.mkv', '.gif']
ESTENSIONI_SUPPORTATE = ESTENSIONI_IMMAGINI + ESTENSIONI_VIDEO


# --- Reverse geocoding OFFLINE (coordinate GPS -> nome del luogo) ---
# reverse_geocoder usa un dataset di città incluso nel pacchetto: NESSUN accesso a
# internet (coerente con la natura "tutto in locale" dell'app). L'import è pigro perché
# al primo utilizzo carica il dataset e costruisce un k-d tree (~qualche secondo).
def ottieni_nome_luogo(latitudine, longitudine):
    """Converte coordinate GPS nella stringa 'Città, Regione, Nazione' (offline).
    Restituisce None se le coordinate mancano o il geocoding fallisce."""
    if latitudine is None or longitudine is None:
        return None
    try:
        import reverse_geocoder as rg
        # mode=1 = single-thread: evita il multiprocessing, problematico su Windows/Streamlit.
        r = rg.search([(float(latitudine), float(longitudine))], mode=1)[0]
        parti = [r.get("name"), r.get("admin1"), r.get("cc")]
        return ", ".join(p for p in parti if p) or None
    except Exception as errore:
        print(f"Reverse geocoding fallito per ({latitudine}, {longitudine}): {errore}")
        return None


def geocodifica_luoghi_mancanti():
    """Backfill una tantum: popola location_name per gli elementi già in archivio che
    hanno coordinate GPS ma nessun nome di luogo. Restituisce il numero aggiornati."""
    connessione = database.ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute(
        "SELECT id, location_lat, location_lon FROM media_items "
        "WHERE location_lat IS NOT NULL AND location_lon IS NOT NULL "
        "AND (location_name IS NULL OR location_name = '')"
    )
    righe = cursore.fetchall()
    if not righe:
        connessione.close()
        return 0

    aggiornati = 0
    try:
        import reverse_geocoder as rg
        coordinate = [(float(r[1]), float(r[2])) for r in righe]
        risultati = rg.search(coordinate, mode=1)  # una sola ricerca in blocco per tutti
        for (id_media, _, _), ris in zip(righe, risultati):
            parti = [ris.get("name"), ris.get("admin1"), ris.get("cc")]
            nome = ", ".join(p for p in parti if p)
            if nome:
                cursore.execute("UPDATE media_items SET location_name = ? WHERE id = ?", (nome, id_media))
                aggiornati += 1
        connessione.commit()
    except Exception as errore:
        print(f"Backfill reverse geocoding fallito: {errore}")
    finally:
        connessione.close()
    return aggiornati

def calcola_sha256(percorso_file):
    """Calcola l'hash SHA-256 di un file per identificare i duplicati."""
    hash_sha256 = hashlib.sha256()
    with open(percorso_file, "rb") as file:
        while blocco := file.read(8192):
            hash_sha256.update(blocco)
    return hash_sha256.hexdigest()

def ottieni_gradi_decimali(dati_gps, riferimento):
    """Converte le coordinate GPS da gradi/minuti/secondi (DMS) a gradi decimali."""
    try:
        gradi = float(dati_gps[0][0]) / float(dati_gps[0][1]) if isinstance(dati_gps[0], tuple) else float(dati_gps[0])
        minuti = float(dati_gps[1][0]) / float(dati_gps[1][1]) if isinstance(dati_gps[1], tuple) else float(dati_gps[1])
        secondi = float(dati_gps[2][0]) / float(dati_gps[2][1]) if isinstance(dati_gps[2], tuple) else float(dati_gps[2])
        
        valore_decimale = gradi + (minuti / 60.0) + (secondi / 3600.0)
        if riferimento in ['S', 'W']:
            valore_decimale = -valore_decimale
        return valore_decimale
    except Exception as errore:
        print(f"Errore durante l'analisi delle coordinate GPS: {errore}")
        return None

def estrai_metadati_exif(percorso_immagine):
    """Estrae la data di creazione e le coordinate GPS dai dati EXIF dell'immagine."""
    data_creazione = None
    latitudine = None
    longitudine = None
    
    try:
        with Image.open(percorso_immagine) as img:
            dati_exif = img._getexif()
            if not dati_exif:
                return data_creazione, latitudine, longitudine
                
            informazioni_gps = {}
            for tag_id, valore in dati_exif.items():
                nome_tag = TAGS.get(tag_id, tag_id)
                # DateTimeOriginal (data di scatto) ha priorità su DateTime (ultima
                # modifica): quest'ultimo è solo un ripiego se il primo manca.
                if nome_tag == "DateTimeOriginal" or (nome_tag == "DateTime" and data_creazione is None):
                    # Le date EXIF sono solitamente formattate come "AAAA:MM:GG HH:MM:SS"
                    try:
                        dt = datetime.datetime.strptime(str(valore), "%Y:%m:%d %H:%M:%S")
                        data_creazione = dt.isoformat()
                    except ValueError:
                        pass
                elif nome_tag == "GPSInfo":
                    for tag_gps_id in valore:
                        nome_tag_gps = GPSTAGS.get(tag_gps_id, tag_gps_id)
                        informazioni_gps[nome_tag_gps] = valore[tag_gps_id]
                        
            # Estrae latitudine e longitudine
            if "GPSLatitude" in informazioni_gps and "GPSLatitudeRef" in informazioni_gps and "GPSLongitude" in informazioni_gps and "GPSLongitudeRef" in informazioni_gps:
                latitudine = ottieni_gradi_decimali(informazioni_gps["GPSLatitude"], informazioni_gps["GPSLatitudeRef"])
                longitudine = ottieni_gradi_decimali(informazioni_gps["GPSLongitude"], informazioni_gps["GPSLongitudeRef"])
                
    except Exception as errore:
        print(f"Errore durante la lettura dei dati EXIF per {percorso_immagine}: {errore}")
        
    return data_creazione, latitudine, longitudine

def ottieni_proprieta_video(percorso_video):
    """Estrae larghezza, altezza, FPS, conteggio frame e durata del video usando OpenCV."""
    cattura = cv2.VideoCapture(percorso_video)
    if not cattura.isOpened():
        return None
        
    larghezza = int(cattura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altezza = int(cattura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cattura.get(cv2.CAP_PROP_FPS)
    conteggio_frame = int(cattura.get(cv2.CAP_PROP_FRAME_COUNT))
    durata = conteggio_frame / fps if fps > 0 else 0.0
    
    cattura.release()
    return {
        "larghezza": larghezza,
        "altezza": altezza,
        "fps": fps,
        "conteggio_frame": conteggio_frame,
        "durata": durata
    }

def crea_anteprima(percorso_originale, tipo_media):
    """Genera un'immagine di anteprima (thumbnail) e la salva nella cartella dedicata."""
    nome_file = os.path.basename(percorso_originale)
    nome_senza_est, _ = os.path.splitext(nome_file)
    percorso_anteprima = os.path.join(config.DIR_ANTEPRIME, f"{nome_senza_est}.jpg")
    
    try:
        if tipo_media == 'image':
            with Image.open(percorso_originale) as img_raw:
                # Applica l'orientamento EXIF (le foto verticali da telefono hanno un tag
                # "Orientation": senza questo la miniatura verrebbe salvata ruotata).
                img = ImageOps.exif_transpose(img_raw)
                img.thumbnail((300, 300))
                # JPEG non supporta alfa/palette: converte qualsiasi modalità non-RGB
                # (RGBA, LA, P dei PNG con palette, ecc.) per evitare errori in salvataggio.
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(percorso_anteprima, "JPEG")
        elif tipo_media == 'video':
            cattura = cv2.VideoCapture(percorso_originale)
            if cattura.isOpened():
                # Legge il primo frame per l'anteprima
                letto_correttamente, frame = cattura.read()
                if letto_correttamente:
                    # Converte in RGB ed in formato PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail((300, 300))
                    img.save(percorso_anteprima, "JPEG")
                cattura.release()
        return percorso_anteprima if os.path.exists(percorso_anteprima) else None
    except Exception as errore:
        print(f"Impossibile creare l'anteprima per {percorso_originale}: {errore}")
        return None

def estrai_audio_video(percorso_video):
    """Estrae l'audio in un file WAV temporaneo (16kHz mono) per la trascrizione con Whisper."""
    import imageio_ffmpeg
    try:
        cartella_temporanea = tempfile.gettempdir()
        percorso_audio_temporaneo = os.path.join(cartella_temporanea, f"{os.path.basename(percorso_video)}_temp.wav")
        
        eseguibile_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Comando FFmpeg per estrarre l'audio:
        # -vn: esclude video, -ac 1: mono, -ar 16000: frequenza di campionamento 16kHz, -y: sovrascrive
        comando = [
            eseguibile_ffmpeg, "-y",
            "-i", percorso_video,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            percorso_audio_temporaneo
        ]
        
        # Nasconde la finestra del prompt dei comandi su Windows
        informazioni_avvio = None
        if os.name == 'nt':
            informazioni_avvio = subprocess.STARTUPINFO()
            informazioni_avvio.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        subprocess.run(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=informazioni_avvio,
            check=True
        )
        return percorso_audio_temporaneo
    except Exception as errore:
        print(f"Estrazione audio tramite FFmpeg fallita: {errore}")
        return None

# Categorie comuni per la classificazione zero-shot di CLIP
CATEGORIE = [
    "person", "man", "woman", "child", "family", "group of people",
    "dog", "cat", "bird", "horse", "animal", "pet",
    "car", "motorcycle", "bicycle", "truck", "airplane", "boat", "vehicle",
    "building", "house", "office", "street", "road", "city",
    "nature", "tree", "forest", "flower", "garden", "grass",
    "mountain", "hill", "beach", "sea", "ocean", "river", "lake", "water",
    "sunset", "sunrise", "sky", "cloud", "snow", "ice",
    "food", "beverage", "fruit", "dinner", "interior", "living room", "kitchen",
    "document", "text", "book", "computer", "phone", "sports", "art"
]

_embedding_categorie_precomputati = None

def _embedding_categorie(qwen):
    """Embedding testuali delle categorie (senza istruzione: sono 'documenti')."""
    global _embedding_categorie_precomputati
    if _embedding_categorie_precomputati is None:
        _embedding_categorie_precomputati = [
            (cat, qwen.ottieni_embedding_testo(f"a photo of a {cat}", con_istruzione=False))
            for cat in CATEGORIE
        ]
    return _embedding_categorie_precomputati

def classifica_tag(embedding_immagine, soglia=None, max_tag=None):
    """Tag zero-shot: softmax(coseni * config.SCALA_LOGIT_TAG) sulle categorie comuni."""
    if soglia is None:
        soglia = config.SOGLIA_PROBABILITA_TAG
    if max_tag is None:
        max_tag = config.MASSIMO_TAG_PER_IMMAGINE
    embedding_cat = _embedding_categorie(gestore.ottieni_qwen())
    similarita = np.array([float(np.dot(embedding_immagine, e)) for _, e in embedding_cat])
    logits = similarita * config.SCALA_LOGIT_TAG
    exp_logits = np.exp(logits - np.max(logits))
    probabilita = exp_logits / np.sum(exp_logits)
    coppie = [(cat, p) for (cat, _), p in zip(embedding_cat, probabilita) if p >= soglia]
    coppie.sort(key=lambda x: x[1], reverse=True)
    return [cat for cat, _ in coppie[:max_tag]]

def registra_file(percorso_origine, hash_precalcolato=None):
    """
    Fase 1 dell'import: copia il file nell'archivio (nome-hash) e lo registra
    nel database in stato 'non elaborato' (processed=0), SENZA avviare la
    pipeline AI. Registrare subito tutto un lotto rende il conteggio
    "file in coda" della dashboard fedele a ciò che resta da elaborare.

    `hash_precalcolato` permette di riutilizzare un hash SHA-256 gia' calcolato
    dal chiamante, evitando di rileggere l'intero file una seconda volta.

    Ritorna la tripla (id_media, percorso_archiviato, tipo_media).
    """
    if not os.path.exists(percorso_origine):
        raise FileNotFoundError(f"Il file sorgente {percorso_origine} non esiste.")

    # 1. Verifica il tipo di file
    nome_file = os.path.basename(percorso_origine)
    _, estensione = os.path.splitext(nome_file)
    estensione = estensione.lower()

    if estensione in ESTENSIONI_IMMAGINI:
        tipo_media = 'image'
    elif estensione in ESTENSIONI_VIDEO:
        tipo_media = 'video'
    else:
        raise ValueError(f"Formato file non supportato: {estensione}")

    # 2. Calcola l'hash univoco del file (o riutilizza quello fornito)
    hash_file = hash_precalcolato or calcola_sha256(percorso_origine)

    # 3. Copia il file nell'archivio usando il suo hash come nome (evita collisioni)
    nome_file_archiviato = f"{hash_file}{estensione}"
    percorso_archiviato = os.path.join(config.DIR_ARCHIVIO, nome_file_archiviato)

    if not os.path.exists(percorso_archiviato):
        shutil.copy2(percorso_origine, percorso_archiviato)

    dimensione_file = os.path.getsize(percorso_archiviato)

    # 4. Registra l'elemento nel database (in stato non elaborato = 0)
    id_media = database.aggiungi_elemento_multimediale(
        percorso_file=percorso_archiviato,
        nome_file=nome_file,
        tipo_media=tipo_media,
        dimensione_file=dimensione_file,
        hash_file=hash_file
    )
    return id_media, percorso_archiviato, tipo_media

def _prepara_media(elemento):
    """Stadio veloce senza AI: miniatura, EXIF/GPS, estrazione frame. Porta
    processed a 1 (visibile in galleria) o -1. Idempotente: azzera i frame parziali."""
    id_media, percorso, tipo_media = elemento["id"], elemento["file_path"], elemento["media_type"]

    # riesecuzione dopo un crash: elimina frame parziali
    connessione = database.ottieni_connessione()
    connessione.execute("DELETE FROM media_frames WHERE media_id = ?", (id_media,))
    connessione.commit()
    connessione.close()

    crea_anteprima(percorso, tipo_media)

    if tipo_media == "image":
        pil_img = ImageOps.exif_transpose(Image.open(percorso)).convert("RGB")
        larghezza, altezza = pil_img.size
        data_creazione, lat, lon = estrai_metadati_exif(percorso)
        if data_creazione is None:
            data_creazione = datetime.datetime.fromtimestamp(os.stat(percorso).st_mtime).isoformat()
        nome_localita = ottieni_nome_luogo(lat, lon)
        database.crea_frame(id_media, 0, 0.0, percorso)
        # le immagini non hanno audio: lo stadio trascrizione e' gia' "fatto"
        database.imposta_stato_stadio(id_media, "stato_trascrizione", 1)
        # processed=1 per ultimo: un crash non deve lasciare un'immagine senza frame
        database.aggiorna_stato_elaborazione(
            id_media=id_media, data_creazione=data_creazione, latitudine=lat,
            longitudine=lon, nome_localita=nome_localita,
            larghezza=larghezza, altezza=altezza, stato_elaborazione=1)
    else:  # video
        proprieta = ottieni_proprieta_video(percorso)
        if not proprieta:
            raise ValueError("Impossibile leggere i metadati del video.")
        data_creazione = datetime.datetime.fromtimestamp(os.stat(percorso).st_mtime).isoformat()
        # estrazione frame a intervalli regolari (identica alla v0.6, senza AI)
        cattura = cv2.VideoCapture(percorso)
        if not cattura.isOpened():
            raise ValueError("Impossibile aprire il file video.")
        fps = proprieta["fps"]
        intervallo = int(config.INTERVALLO_FRAME_VIDEO * fps) if fps > 0 else 90
        intervallo = intervallo if intervallo > 0 else 90
        indice_frame, conteggio_estratti = 0, 0
        while cattura.isOpened():
            letto, frame = cattura.read()
            if not letto:
                break
            if indice_frame % intervallo == 0:
                timestamp = indice_frame / fps if fps > 0 else 0.0
                percorso_frame = os.path.join(config.DIR_FRAME, f"{id_media}_frame_{conteggio_estratti}.jpg")
                cv2.imwrite(percorso_frame, frame)
                database.crea_frame(id_media, conteggio_estratti, timestamp, percorso_frame)
                conteggio_estratti += 1
            indice_frame += 1
        cattura.release()
        # Le GIF non hanno traccia audio: trascrizione gia' "fatta", come per le immagini
        # (evita di caricare Whisper e lanciare ffmpeg su un file senza audio).
        if os.path.splitext(percorso)[1].lower() == ".gif":
            database.imposta_stato_stadio(id_media, "stato_trascrizione", 1)
        database.aggiorna_stato_elaborazione(
            id_media=id_media, data_creazione=data_creazione,
            larghezza=proprieta["larghezza"], altezza=proprieta["altezza"],
            durata=proprieta["durata"], stato_elaborazione=1)

def _stadio_embedding(elemento):
    """Embedding Qwen + tag per ogni frame senza embedding (riprende dai mancanti)."""
    qwen = gestore.ottieni_qwen()
    for frame in database.ottieni_frame_di_media(elemento["id"]):
        if frame["clip_embedding_presente"]:
            continue
        emb = qwen.ottieni_embedding_immagine(frame["image_path"])
        tags = classifica_tag(emb)
        database.aggiorna_embedding_frame(frame["id"], elemento["id"], emb, tags)

def _stadio_volti(elemento):
    """MTCNN+FaceNet su ogni frame; ogni volto viene assegnato a una persona."""
    import persone
    face_rec = gestore.ottieni_volti()
    database.elimina_volti_di_media(elemento["id"])  # idempotenza su rieseguzione
    for frame in database.ottieni_frame_di_media(elemento["id"]):
        pil_img = ImageOps.exif_transpose(Image.open(frame["image_path"])).convert("RGB")
        for idx, dati_volto in enumerate(face_rec.rileva_e_codifica_volti(pil_img)):
            percorso_volto = os.path.join(
                config.DIR_VOLTI, f"{elemento['id']}_fr_{frame['id']}_face_{idx}.jpg")
            dati_volto["crop"].save(percorso_volto, "JPEG")
            id_volto = database.aggiungi_volto(
                id_media=elemento["id"], id_frame=frame["id"],
                percorso_ritaglio=percorso_volto,
                embedding=dati_volto["embedding"], riquadro=dati_volto["bbox"])
            persone.assegna_persona(dati_volto["embedding"], id_volto)

def _stadio_trascrizione(elemento):
    """Solo video: estrazione audio + Whisper."""
    whisper = gestore.ottieni_whisper()
    trascrizione = None
    percorso_audio = estrai_audio_video(elemento["file_path"])
    if percorso_audio:
        try:
            trascrizione = whisper.trascrivi(percorso_audio)
        finally:
            if os.path.exists(percorso_audio):
                os.remove(percorso_audio)
    database.aggiorna_stato_elaborazione(id_media=elemento["id"], trascrizione=trascrizione,
                                         stato_elaborazione=1)

def aggiungi_e_elabora_file(percorso_origine, hash_precalcolato=None):
    """Importa un file: copia in archivio + record nel DB. L'elaborazione AI
    avviene in background tramite la coda a stadi (avvia_lavoratore)."""
    id_media, percorso_archiviato, tipo_media = registra_file(percorso_origine, hash_precalcolato)
    avvia_lavoratore()
    return id_media, True

# ---------------------------------------------------------------------------
# Lavoratore in background (coda persistente su SQLite)
#
# La coda NON e' una struttura in memoria: e' la query database.prossimo_media_in_coda()
# sulle colonne di stadio. Un riavvio riprende esattamente da dove si era rimasti.
# Il thread daemon lavora un elemento alla volta e controlla la pausa tra gli stadi.
# ponytail: un solo thread lavoratore; processo separato solo se i rerun di
# Streamlit dovessero interferire.
# ---------------------------------------------------------------------------

_lavoratore = None
_lock_lavoratore = threading.Lock()
_sveglia = threading.Event()
_stato_runtime = {"in_corso": None, "durate": []}  # durate: ultimi secondi/elemento per l'ETA

def avvia_lavoratore():
    """Avvia (una sola volta per processo) il thread della coda e lo sveglia."""
    global _lavoratore
    with _lock_lavoratore:
        if _lavoratore is None or not _lavoratore.is_alive():
            _lavoratore = threading.Thread(target=_ciclo_lavoratore, daemon=True,
                                           name="deepsight-coda")
            _lavoratore.start()
    _sveglia.set()

def metti_in_pausa():
    database.scrivi_impostazione("coda_in_pausa", "1")

def riprova_falliti():
    """Rimette in coda gli elementi/stadi falliti (-1 -> 0). Ritorna quanti."""
    connessione = database.ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("UPDATE media_items SET processed = 0 WHERE processed = -1")
    n = cursore.rowcount
    for colonna in ("stato_embedding", "stato_volti", "stato_trascrizione"):
        cursore.execute(f"UPDATE media_items SET {colonna} = 0 WHERE {colonna} = -1")
        n += cursore.rowcount
    connessione.commit()
    connessione.close()
    _sveglia.set()
    return n

def riprendi():
    database.scrivi_impostazione("coda_in_pausa", "0")
    _sveglia.set()

def stato_coda():
    """Fotografia per la UI: conteggi, elemento in corso, pausa, ETA stimato."""
    conteggi = database.conteggio_coda()
    # File distinti, non somma degli stadi: 10 foto = "10 rimanenti", non 30.
    rimanenti = conteggi["file_in_coda"]
    durate = _stato_runtime["durate"]
    # ETA approssimativa: media delle ultime iterazioni x file rimanenti (un file
    # appena registrato passa 2 iterazioni: preparazione + stadi AI, quindi sottostima un po')
    eta = (sum(durate) / len(durate)) * rimanenti if durate and rimanenti else None
    return {**conteggi, "rimanenti": rimanenti, "in_corso": _stato_runtime["in_corso"],
            "in_pausa": database.leggi_impostazione("coda_in_pausa", "0") == "1",
            "eta_secondi": eta}

_STADI = (("stato_embedding", "embedding", _stadio_embedding),
          ("stato_volti", "volti", _stadio_volti),
          ("stato_trascrizione", "trascrizione", _stadio_trascrizione))

def _ciclo_lavoratore():
    import time
    while True:
        if database.leggi_impostazione("coda_in_pausa", "0") == "1":
            _sveglia.clear()
            _sveglia.wait(timeout=2.0)
            continue
        elemento = database.prossimo_media_in_coda()
        if elemento is None:
            _sveglia.clear()
            _sveglia.wait(timeout=2.0)
            continue

        inizio = time.monotonic()
        try:
            if elemento["processed"] == 0:
                _stato_runtime["in_corso"] = f"{elemento['filename']} · preparazione"
                _prepara_media(elemento)
            else:
                for colonna, etichetta, stadio in _STADI:
                    if elemento[colonna] != 0:
                        continue
                    if database.leggi_impostazione("coda_in_pausa", "0") == "1":
                        break  # pausa tra gli stadi: l'elemento restera' in coda
                    _stato_runtime["in_corso"] = f"{elemento['filename']} · {etichetta}"
                    try:
                        stadio(elemento)
                        database.imposta_stato_stadio(elemento["id"], colonna, 1)
                    except Exception as errore:
                        print(f"Stadio {colonna} fallito per {elemento['filename']}: {errore}")
                        database.imposta_stato_stadio(elemento["id"], colonna, -1)
        except Exception as errore:
            print(f"Preparazione fallita per {elemento['filename']}: {errore}")
            database.aggiorna_stato_elaborazione(id_media=elemento["id"], stato_elaborazione=-1)
        finally:
            _stato_runtime["in_corso"] = None
            durate = _stato_runtime["durate"]
            durate.append(time.monotonic() - inizio)
            del durate[:-20]  # media mobile sugli ultimi 20 elementi

def scansiona_cartella_condivisa(percorso_cartella):
    """Scansiona una cartella condivisa per individuare nuovi file da elaborare ed importare."""
    if not os.path.exists(percorso_cartella):
        return 0, 0
        
    conteggio_successi = 0
    conteggio_fallimenti = 0

    for nome_file in os.listdir(percorso_cartella):
        percorso_file = os.path.join(percorso_cartella, nome_file)
        if os.path.isdir(percorso_file):
            continue

        _, estensione = os.path.splitext(nome_file)
        if estensione.lower() in ESTENSIONI_SUPPORTATE:
            try:
                # Controlla l'hash per verificare se è già nel database prima di caricare i modelli pesanti
                hash_file = calcola_sha256(percorso_file)
                connessione = database.ottieni_connessione()
                cursore = connessione.cursor()
                cursore.execute("SELECT id FROM media_items WHERE file_hash = ? AND processed = 1", (hash_file,))
                gia_presente = cursore.fetchone()
                connessione.close()
                
                if gia_presente:
                    continue  # Già elaborato in precedenza
                    
                print(f"Elaborazione di un nuovo file trovato in cartella condivisa: {nome_file}")
                aggiungi_e_elabora_file(percorso_file)
                conteggio_successi += 1
            except Exception as errore:
                print(f"Elaborazione fallita per il file {nome_file} in cartella condivisa: {errore}")
                conteggio_fallimenti += 1
                
    return conteggio_successi, conteggio_fallimenti


# ---------------------------------------------------------------------------
# Gestione dei file "intrusi" nella cartella archivio
#
# L'archivio (config.DIR_ARCHIVIO) e' interamente gestito dall'app: ogni file
# legittimo viene copiato con il proprio hash SHA-256 come nome e registrato nel
# database. Un file "intruso" e' quindi un file presente in data/archive che NON
# risulta registrato nel database, tipicamente perche' copiato a mano dall'utente
# e mai indicizzato dalla pipeline AI.
# ---------------------------------------------------------------------------

def _cartella_quarantena():
    """Restituisce (creandola se serve) la cartella dove isolare i file estranei."""
    os.makedirs(config.DIR_QUARANTENA, exist_ok=True)
    return config.DIR_QUARANTENA

def _sposta_in_quarantena(percorso_file):
    """Sposta un file nella cartella di quarantena, evitando sovrascritture."""
    try:
        destinazione = os.path.join(_cartella_quarantena(), os.path.basename(percorso_file))
        base, estensione = os.path.splitext(destinazione)
        contatore = 1
        while os.path.exists(destinazione):
            destinazione = f"{base}_{contatore}{estensione}"
            contatore += 1
        shutil.move(percorso_file, destinazione)
        return True
    except Exception as errore:
        print(f"Impossibile spostare in quarantena {percorso_file}: {errore}")
        return False

def trova_file_intrusi():
    """Elenca i file presenti in data/archive che non sono registrati nel database.

    Sono i file aggiunti manualmente dall'utente: non hanno tag, volti, testo o
    embedding e restano fuori sincrono rispetto all'archivio indicizzato.
    """
    if not os.path.isdir(config.DIR_ARCHIVIO):
        return []

    percorsi_registrati = database.ottieni_percorsi_archiviati()
    intrusi = []
    for nome_file in os.listdir(config.DIR_ARCHIVIO):
        percorso = os.path.join(config.DIR_ARCHIVIO, nome_file)
        if not os.path.isfile(percorso):
            continue
        if os.path.normcase(os.path.abspath(percorso)) not in percorsi_registrati:
            intrusi.append(percorso)
    return intrusi

def importa_file_intrusi():
    """Adotta i file intrusi trovati in archivio.

    Per ogni file supportato: ne calcola l'hash, lo registra e lo elabora tramite
    la pipeline standard (che lo salva con il nome-hash corretto); l'eventuale copia
    con il nome originale "sbagliato" viene poi rimossa. I formati non supportati
    vengono spostati in quarantena.

    Ritorna la tripla (importati, spostati_in_quarantena, falliti).
    """
    intrusi = trova_file_intrusi()
    importati = 0
    in_quarantena = 0
    falliti = 0

    for percorso in intrusi:
        _, estensione = os.path.splitext(percorso)
        estensione = estensione.lower()

        # Formati non multimediali: non indicizzabili, si isolano in quarantena
        if estensione not in ESTENSIONI_SUPPORTATE:
            if _sposta_in_quarantena(percorso):
                in_quarantena += 1
            else:
                falliti += 1
            continue

        percorso_hash = None  # nome-hash di destinazione, definito solo dopo il calcolo
        try:
            # Calcola l'hash una sola volta e determina il nome-hash di destinazione
            hash_file = calcola_sha256(percorso)
            percorso_hash = os.path.join(config.DIR_ARCHIVIO, f"{hash_file}{estensione}")

            aggiungi_e_elabora_file(percorso, hash_precalcolato=hash_file)
            importati += 1
        except Exception as errore:
            print(f"Import del file intruso {percorso} fallito: {errore}")
            falliti += 1
            continue
        finally:
            # Rimuove l'originale con nome diverso se ora esiste la copia con nome-hash
            try:
                if percorso_hash is not None:
                    stesso_file = os.path.normcase(os.path.abspath(percorso_hash)) == \
                                  os.path.normcase(os.path.abspath(percorso))
                    if not stesso_file and os.path.exists(percorso_hash) and os.path.exists(percorso):
                        os.remove(percorso)
            except Exception as errore:
                print(f"Impossibile rimuovere l'originale intruso {percorso}: {errore}")

    return importati, in_quarantena, falliti

def trova_record_orfani():
    """Elenca i record del database il cui file originale non esiste più su disco
    (rimosso a mano dall'archivio): il caso speculare dei file "intrusi".

    Restano nel DB con embedding, tag e volti, quindi continuano a comparire in
    ricerca e galleria pur senza file. Ritorna una lista di dizionari.
    """
    connessione = database.ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("SELECT id, filename, file_path, media_type, processed FROM media_items")
    righe = cursore.fetchall()
    connessione.close()

    orfani = []
    for id_media, nome_file, percorso, tipo_media, stato in righe:
        if not percorso or not os.path.exists(percorso):
            orfani.append({
                "id": id_media, "filename": nome_file, "file_path": percorso,
                "media_type": tipo_media, "processed": stato,
            })
    return orfani

def rimuovi_record_orfani():
    """Rimuove dall'indice tutti i record orfani: righe del DB, vettori Chroma e
    file residui ancora su disco (miniature, frame estratti, ritagli dei volti).
    Ritorna il numero di record rimossi."""
    orfani = trova_record_orfani()
    for orfano in orfani:
        database.elimina_elemento_multimediale(orfano["id"])
    return len(orfani)

def sposta_file_intrusi_in_quarantena():
    """Sposta tutti i file intrusi nella cartella di quarantena senza indicizzarli.

    Ritorna il numero di file effettivamente spostati.
    """
    intrusi = trova_file_intrusi()
    spostati = 0
    for percorso in intrusi:
        if _sposta_in_quarantena(percorso):
            spostati += 1
    return spostati

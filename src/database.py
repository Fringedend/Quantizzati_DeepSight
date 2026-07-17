import sqlite3
import numpy as np
import os
import json
import config

# Inizializza i due archivi vettoriali (ChromaDB): collezioni separate perche' gli
# id dei frame e dei volti sono sequenze AUTOINCREMENT distinte e colliderebbero.
_store_frame = None
_store_volti = None
try:
    # Import dentro il try: se chromadb non importa (es. sqlite3 di sistema troppo
    # vecchio), gli store restano None e la ricerca degrada alla scansione lineare.
    from chroma_store import ChromaStore
    _store_frame = ChromaStore(nome_collezione="qwen_frames")
    _store_volti = ChromaStore(nome_collezione="faces")
except Exception as errore:
    print(f"Errore durante l'inizializzazione degli archivi vettoriali: {errore}")

def ottieni_connessione():
    """Restituisce una connessione al database SQLite locale."""
    connessione = sqlite3.connect(config.PERCORSO_DB)
    connessione.execute("PRAGMA foreign_keys = ON;")
    # WAL fa coesistere UN lettore e UNO scrittore, ma NON serializza due scrittori:
    # con worker in background + UI che scrivono, il secondo scrittore fallirebbe
    # subito con "database is locked". busy_timeout lo fa aspettare il lock invece.
    connessione.execute("PRAGMA journal_mode=WAL;")
    connessione.execute(f"PRAGMA busy_timeout={config.SQLITE_BUSY_TIMEOUT_MS};")
    return connessione

def inizializza_db():
    """Crea le tabelle del database e gli indici se non esistono già."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    
    # Tabella degli elementi multimediali (immagini e video)
    cursore.execute("""
    CREATE TABLE IF NOT EXISTS media_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        filename TEXT,
        media_type TEXT, -- 'image' o 'video'
        file_size INTEGER,
        creation_date TEXT,
        location_lat REAL,
        location_lon REAL,
        location_name TEXT,
        width INTEGER,
        height INTEGER,
        duration REAL, -- NULL per le immagini
        transcription TEXT, -- Trascrizione vocale (Speech-to-Text)
        file_hash TEXT UNIQUE, -- Hash SHA-256 per evitare duplicati
        processed INTEGER DEFAULT 0 -- 0=non elaborato, 1=elaborato, -1=fallito
    );
    """)
    
    # Tabella per i singoli frame dei video (o frame singolo per le immagini)
    cursore.execute("""
    CREATE TABLE IF NOT EXISTS media_frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER,
        frame_index INTEGER,
        timestamp_seconds REAL,
        image_path TEXT,
        ocr_text TEXT,
        objects TEXT, -- Lista di tag in formato JSON o testo separato da virgola
        clip_embedding BLOB, -- Array NumPy salvato in formato binario (BLOB)
        FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE
    );
    """)
    
    # Tabella per i volti rilevati negli elementi multimediali
    cursore.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER,
        frame_id INTEGER, -- NULL per le immagini, punta a media_frames per i video
        crop_path TEXT,
        embedding BLOB, -- Array NumPy del volto in formato binario (BLOB)
        box_x1 REAL,
        box_y1 REAL,
        box_x2 REAL,
        box_y2 REAL,
        FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE,
        FOREIGN KEY(frame_id) REFERENCES media_frames(id) ON DELETE CASCADE
    );
    """)
    
    # Creazione degli indici per velocizzare le ricerche
    cursore.execute("CREATE INDEX IF NOT EXISTS idx_hash ON media_items(file_hash);")
    cursore.execute("CREATE INDEX IF NOT EXISTS idx_processed ON media_items(processed);")
    cursore.execute("CREATE INDEX IF NOT EXISTS idx_creation_date ON media_items(creation_date);")
    cursore.execute("CREATE INDEX IF NOT EXISTS idx_frames_media ON media_frames(media_id);")
    cursore.execute("CREATE INDEX IF NOT EXISTS idx_faces_media ON faces(media_id);")

    # Tabella chiave/valore per lo stato runtime (es. pausa della coda)
    cursore.execute("""
    CREATE TABLE IF NOT EXISTS impostazioni (
        chiave TEXT PRIMARY KEY,
        valore TEXT
    );
    """)

    # Tabella delle persone (raggruppamento dei volti per il Face Database)
    cursore.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    );
    """)

    connessione.commit()
    _migra_schema_v07(connessione)
    connessione.close()
    _sincronizza_indici_vettoriali()

def _migra_schema_v07(connessione):
    """Migrazione v0.7: colonne di stadio su media_items, person_id sui volti,
    azzeramento degli embedding CLIP 512-d (incompatibili con Qwen 2048-d).

    Eseguita UNA volta sola (flag in impostazioni): il backfill degli stadi
    marcherebbe come 'fatti' gli stadi legittimamente pendenti dei media v0.7."""
    cursore = connessione.cursor()
    cursore.execute("SELECT valore FROM impostazioni WHERE chiave = 'migrazione_v07_fatta'")
    if cursore.fetchone():
        return

    colonne_media = {r[1] for r in cursore.execute("PRAGMA table_info(media_items)")}
    for colonna in ("stato_embedding", "stato_volti", "stato_trascrizione"):
        if colonna not in colonne_media:
            cursore.execute(f"ALTER TABLE media_items ADD COLUMN {colonna} INTEGER DEFAULT 0")

    colonne_volti = {r[1] for r in cursore.execute("PRAGMA table_info(faces)")}
    if "person_id" not in colonne_volti:
        cursore.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER REFERENCES persons(id)")
        cursore.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")

    # Embedding della vecchia era CLIP (512 float32 = 2048 byte): vanno rigenerati.
    # Si azzerano i BLOB e si rimettono in coda i media interessati.
    cursore.execute("SELECT COUNT(*) FROM media_frames WHERE clip_embedding IS NOT NULL "
                    "AND LENGTH(clip_embedding) != ?", (config.DIM_EMBEDDING_QWEN * 4,))
    if cursore.fetchone()[0] > 0:
        print("Migrazione: trovati embedding CLIP 512-d, verranno rigenerati con Qwen.")
        cursore.execute("""
            UPDATE media_items SET stato_embedding = 0 WHERE id IN (
                SELECT DISTINCT media_id FROM media_frames
                WHERE clip_embedding IS NOT NULL AND LENGTH(clip_embedding) != ?)
        """, (config.DIM_EMBEDDING_QWEN * 4,))
        cursore.execute("UPDATE media_frames SET clip_embedding = NULL "
                        "WHERE clip_embedding IS NOT NULL AND LENGTH(clip_embedding) != ?",
                        (config.DIM_EMBEDDING_QWEN * 4,))
    # Archivi pre-v0.7 gia' elaborati: volti e trascrizione esistono gia'
    cursore.execute("UPDATE media_items SET stato_volti = 1, stato_trascrizione = 1 "
                    "WHERE processed = 1 AND stato_volti = 0 AND id IN "
                    "(SELECT DISTINCT media_id FROM media_frames)")
    cursore.execute("INSERT INTO impostazioni (chiave, valore) VALUES ('migrazione_v07_fatta', '1')")
    connessione.commit()

    # Gli archivi pre-v0.7 hanno gia' volti rilevati ma nessuna persona (person_id
    # inesistente prima di questa migrazione): li si raggruppa subito, altrimenti
    # la scheda Persone risulta vuota nonostante l'archivio sia pieno di volti.
    cursore.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NULL")
    if cursore.fetchone()[0] > 0:
        connessione.commit()  # il re-cluster apre connessioni proprie
        import persone
        n = persone.ricalcola_tutti_cluster()
        print(f"Migrazione: {n} persone create dai volti esistenti.")

def leggi_impostazione(chiave, default=None):
    connessione = ottieni_connessione()
    riga = connessione.execute("SELECT valore FROM impostazioni WHERE chiave = ?", (chiave,)).fetchone()
    connessione.close()
    return riga[0] if riga else default

def scrivi_impostazione(chiave, valore):
    connessione = ottieni_connessione()
    connessione.execute("INSERT INTO impostazioni (chiave, valore) VALUES (?, ?) "
                        "ON CONFLICT(chiave) DO UPDATE SET valore = excluded.valore",
                        (chiave, str(valore)))
    connessione.commit()
    connessione.close()

def crea_frame(id_media, indice_frame, secondi_timestamp, percorso_immagine):
    """Registra un frame SENZA embedding (lo stadio embedding lo riempira' poi)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("""
    INSERT INTO media_frames (media_id, frame_index, timestamp_seconds, image_path, ocr_text, objects, clip_embedding)
    VALUES (?, ?, ?, ?, '', '[]', NULL)
    """, (id_media, indice_frame, secondi_timestamp, percorso_immagine))
    id_frame = cursore.lastrowid
    connessione.commit()
    connessione.close()
    return id_frame

def aggiorna_embedding_frame(id_frame, id_media, embedding, oggetti):
    """Scrive embedding Qwen (colonna storica 'clip_embedding') e tag, e aggiorna Chroma."""
    connessione = ottieni_connessione()
    connessione.execute("UPDATE media_frames SET clip_embedding = ?, objects = ? WHERE id = ?",
                        (serializza_vettore(embedding), json.dumps(list(oggetti)), id_frame))
    connessione.commit()
    connessione.close()
    if _store_frame is not None:
        try:
            _store_frame.aggiungi_o_aggiorna([id_frame], [embedding], [{"media_id": id_media}])
        except Exception as errore:
            print(f"Errore inserimento vettoriale frame {id_frame}: {errore}")

def ottieni_frame_di_media(id_media):
    connessione = ottieni_connessione()
    righe = connessione.execute(
        "SELECT id, image_path, clip_embedding IS NOT NULL FROM media_frames "
        "WHERE media_id = ? ORDER BY frame_index", (id_media,)).fetchall()
    connessione.close()
    return [{"id": r[0], "image_path": r[1], "clip_embedding_presente": bool(r[2])} for r in righe]

def imposta_stato_stadio(id_media, colonna, valore):
    assert colonna in ("stato_embedding", "stato_volti", "stato_trascrizione"), colonna
    connessione = ottieni_connessione()
    connessione.execute(f"UPDATE media_items SET {colonna} = ? WHERE id = ?", (valore, id_media))
    connessione.commit()
    connessione.close()

def prossimo_media_in_coda():
    """Prossimo elemento da lavorare: prima le preparazioni (processed=0, veloci:
    miniatura+EXIF+frame), poi gli stadi AI pendenti. None se la coda e' vuota."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("SELECT * FROM media_items WHERE processed = 0 ORDER BY id LIMIT 1")
    riga = cursore.fetchone()
    if riga is None:
        cursore.execute("""SELECT * FROM media_items WHERE processed = 1
            AND (stato_embedding = 0 OR stato_volti = 0 OR stato_trascrizione = 0)
            ORDER BY id LIMIT 1""")
        riga = cursore.fetchone()
    colonne = [c[0] for c in cursore.description] if riga else []
    connessione.close()
    return dict(zip(colonne, riga)) if riga else None

def conteggio_coda():
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    def _conta(sql):
        cursore.execute(sql)
        return cursore.fetchone()[0]
    conteggi = {
        "da_preparare": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 0"),
        "embedding": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 1 AND stato_embedding = 0"),
        "volti": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 1 AND stato_volti = 0"),
        "trascrizione": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 1 AND stato_trascrizione = 0"),
        "falliti": _conta("SELECT COUNT(*) FROM media_items WHERE processed = -1 "
                          "OR stato_embedding = -1 OR stato_volti = -1 OR stato_trascrizione = -1"),
        # File distinti ancora in coda (un file con 3 stadi pendenti conta 1, non 3):
        # e' il numero che la UI mostra come "rimanenti".
        "file_in_coda": _conta("SELECT COUNT(*) FROM media_items WHERE processed = 0 "
                               "OR (processed = 1 AND (stato_embedding = 0 OR stato_volti = 0 "
                               "OR stato_trascrizione = 0))"),
    }
    connessione.close()
    return conteggi

def elimina_volti_di_media(id_media):
    """Rimuove volti (righe, ritagli su disco, vettori Chroma) di un media: rende
    idempotente la riesecuzione dello stadio volti dopo un fallimento."""
    connessione = ottieni_connessione()
    righe = connessione.execute("SELECT id, crop_path FROM faces WHERE media_id = ?", (id_media,)).fetchall()
    connessione.execute("DELETE FROM faces WHERE media_id = ?", (id_media,))
    connessione.commit()
    connessione.close()
    if _store_volti is not None and righe:
        try:
            _store_volti.elimina([r[0] for r in righe])
        except Exception as errore:
            print(f"Errore rimozione vettori volto per media {id_media}: {errore}")
    for _, percorso in righe:
        if percorso and os.path.exists(percorso):
            try:
                os.remove(percorso)
            except OSError:
                pass

def crea_persona():
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("INSERT INTO persons (name) VALUES (NULL)")
    id_persona = cursore.lastrowid
    connessione.commit()
    connessione.close()
    return id_persona

def crea_persona_con_volto(id_volto):
    """Crea una persona e le assegna il volto NELLA STESSA transazione: una
    persona non è mai visibile senza volti, così la potatura di ottieni_persone
    non può eliminarla tra creazione e assegnazione (race con il worker)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("INSERT INTO persons (name) VALUES (NULL)")
    id_persona = cursore.lastrowid
    cursore.execute("UPDATE faces SET person_id = ? WHERE id = ?", (id_persona, id_volto))
    connessione.commit()
    connessione.close()
    return id_persona

def assegna_volto_a_persona(id_volto, id_persona):
    connessione = ottieni_connessione()
    connessione.execute("UPDATE faces SET person_id = ? WHERE id = ?", (id_persona, id_volto))
    connessione.commit()
    connessione.close()

def ottieni_embedding_volti_per_persona():
    """{id_persona: [embedding, ...]} per il calcolo dei centroidi (persone.py)."""
    connessione = ottieni_connessione()
    righe = connessione.execute(
        "SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL").fetchall()
    connessione.close()
    per_persona = {}
    for id_persona, blob in righe:
        emb = deserializza_vettore(blob)
        if emb is not None:
            per_persona.setdefault(id_persona, []).append(emb)
    return per_persona

def ottieni_persone():
    """Elenco persone con conteggi e ritaglio rappresentativo. Elimina le persone
    rimaste senza volti (es. dopo la cancellazione dei media)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("DELETE FROM persons WHERE id NOT IN "
                    "(SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)")
    connessione.commit()
    cursore.execute("""
        SELECT p.id, p.name, COUNT(f.id), COUNT(DISTINCT f.media_id), MIN(f.crop_path)
        FROM persons p JOIN faces f ON f.person_id = p.id
        GROUP BY p.id, p.name ORDER BY COUNT(f.id) DESC
    """)
    righe = cursore.fetchall()
    connessione.close()
    return [{"id": r[0], "name": r[1], "n_volti": r[2], "n_media": r[3], "crop_path": r[4]}
            for r in righe]

def ottieni_media_di_persona(id_persona):
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("""
        SELECT DISTINCT m.* FROM media_items m JOIN faces f ON f.media_id = m.id
        WHERE f.person_id = ? ORDER BY m.creation_date DESC
    """, (id_persona,))
    righe = cursore.fetchall()
    colonne = [c[0] for c in cursore.description]
    connessione.close()
    return [dict(zip(colonne, r)) for r in righe]

def rinomina_persona(id_persona, nome):
    connessione = ottieni_connessione()
    connessione.execute("UPDATE persons SET name = ? WHERE id = ?", (nome or None, id_persona))
    connessione.commit()
    connessione.close()

def unisci_persone(id_da, id_a):
    """Sposta tutti i volti di id_da su id_a (il nome di id_a prevale) ed elimina id_da."""
    connessione = ottieni_connessione()
    connessione.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (id_a, id_da))
    connessione.execute("DELETE FROM persons WHERE id = ?", (id_da,))
    connessione.commit()
    connessione.close()

def azzera_persone():
    """Scollega tutti i volti e svuota persons (per il re-clustering completo)."""
    connessione = ottieni_connessione()
    connessione.execute("UPDATE faces SET person_id = NULL")
    connessione.execute("DELETE FROM persons")
    connessione.commit()
    connessione.close()

def serializza_vettore(vettore):
    """Converte un array NumPy in byte binari per il salvataggio nel database SQL (BLOB)."""
    if vettore is None:
        return None
    return vettore.astype(np.float32).tobytes()

def deserializza_vettore(blob_dati):
    """Converte una sequenza di byte (BLOB) di nuovo in un array NumPy."""
    if blob_dati is None:
        return None
    return np.frombuffer(blob_dati, dtype=np.float32).copy()

def aggiungi_elemento_multimediale(percorso_file, nome_file, tipo_media, dimensione_file, hash_file, 
                                   data_creazione=None, latitudine=None, longitudine=None, 
                                   nome_localita=None, larghezza=None, altezza=None, durata=None):
    """Inserisce un nuovo elemento multimediale nel database in stato 'non elaborato' (0)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    try:
        cursore.execute("""
        INSERT INTO media_items 
        (file_path, filename, media_type, file_size, file_hash, creation_date, location_lat, location_lon, location_name, width, height, duration, processed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (percorso_file, nome_file, tipo_media, dimensione_file, hash_file, data_creazione, latitudine, longitudine, nome_localita, larghezza, altezza, durata))
        id_media = cursore.lastrowid
        connessione.commit()
        return id_media
    except sqlite3.IntegrityError:
        # Se l'elemento esiste già, recupera l'ID dell'elemento esistente tramite l'hash
        cursore.execute("SELECT id FROM media_items WHERE file_hash = ?", (hash_file,))
        riga = cursore.fetchone()
        if riga:
            return riga[0]
        return None
    finally:
        connessione.close()

def aggiorna_stato_elaborazione(id_media, data_creazione=None, latitudine=None, longitudine=None, 
                               nome_localita=None, larghezza=None, altezza=None, durata=None, trascrizione=None, stato_elaborazione=1):
    """Aggiorna i metadati e lo stato di elaborazione di un elemento multimediale."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    
    # Recupera i valori attuali per evitare di sovrascrivere dati validi con valori None
    cursore.execute("SELECT creation_date, location_lat, location_lon, location_name, width, height, duration, transcription FROM media_items WHERE id = ?", (id_media,))
    valori_correnti = cursore.fetchone()
    if valori_correnti:
        val_data, val_lat, val_lon, val_loc, val_w, val_h, val_dur, val_trascr = valori_correnti
        data_creazione = data_creazione if data_creazione is not None else val_data
        latitudine = latitudine if latitudine is not None else val_lat
        longitudine = longitudine if longitudine is not None else val_lon
        nome_localita = nome_localita if nome_localita is not None else val_loc
        larghezza = larghezza if larghezza is not None else val_w
        altezza = altezza if altezza is not None else val_h
        durata = durata if durata is not None else val_dur
        trascrizione = trascrizione if trascrizione is not None else val_trascr

    cursore.execute("""
    UPDATE media_items
    SET creation_date = ?, location_lat = ?, location_lon = ?, location_name = ?, 
        width = ?, height = ?, duration = ?, transcription = ?, processed = ?
    WHERE id = ?
    """, (data_creazione, latitudine, longitudine, nome_localita, larghezza, altezza, durata, trascrizione, stato_elaborazione, id_media))
    connessione.commit()
    connessione.close()
    return None

def aggiungi_volto(id_media, id_frame, percorso_ritaglio, embedding, riquadro):
    """Salva i dati di un volto rilevato e il relativo embedding FaceNet, sincronizzandolo con ChromaDB."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    blob_embedding = serializza_vettore(embedding)
    x1, y1, x2, y2 = riquadro
    
    cursore.execute("""
    INSERT INTO faces (media_id, frame_id, crop_path, embedding, box_x1, box_y1, box_x2, box_y2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (id_media, id_frame, percorso_ritaglio, blob_embedding, x1, y1, x2, y2))
    id_volto = cursore.lastrowid
    
    # Indicizza il vettore FaceNet nella collezione dei volti
    if _store_volti is not None:
        try:
            _store_volti.aggiungi_o_aggiorna([id_volto], [embedding], [{"media_id": id_media}])
        except Exception as errore:
            print(f"Errore durante l'inserimento vettoriale per il volto {id_volto}: {errore}")
            
    connessione.commit()
    connessione.close()
    return id_volto

def ottieni_elemento_multimediale(id_media):
    """Recupera i dettagli di un singolo elemento multimediale tramite il suo ID."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    # Esegue la query SQL
    cursore.execute("SELECT * FROM media_items WHERE id = ?", (id_media,))
    riga = cursore.fetchone()
    connessione.close()
    if riga:
        colonne = [col[0] for col in cursore.description]
        return dict(zip(colonne, riga))
    return None

def ottieni_tutti_elementi_multimediali(solo_elaborati=True):
    """Recupera tutti gli elementi multimediali presenti nel database."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    query = "SELECT * FROM media_items"
    if solo_elaborati:
        query += " WHERE processed = 1"
    cursore.execute(query)
    righe = cursore.fetchall()
    connessione.close()
    
    colonne = [col[0] for col in cursore.description]
    return [dict(zip(colonne, r)) for r in righe]

def ottieni_tag_per_media():
    """Ritorna {media_id: [tag, ...]} unendo i tag (media_frames.objects) di tutti i frame
    di ogni media, deduplicati mantenendo l'ordine di prima comparsa. Una sola query per
    tutta la galleria, invece di un'interrogazione per elemento. I tag stanno sui frame,
    non sui media_items, perciò la galleria (che legge media_items) va arricchita a parte."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("SELECT media_id, objects FROM media_frames "
                    "WHERE objects IS NOT NULL AND objects != ''")
    righe = cursore.fetchall()
    connessione.close()
    tag_per_media = {}
    for media_id, objects in righe:
        try:
            lista = json.loads(objects)
        except Exception:
            lista = [x.strip() for x in objects.split(",") if x.strip()]
        accumulatore = tag_per_media.setdefault(media_id, [])
        for tag in lista:
            if tag not in accumulatore:
                accumulatore.append(tag)
    return tag_per_media

def ottieni_elementi_multimediali(id_media_lista):
    """Recupera più elementi preservando l'ordine degli ID richiesti."""
    ids = list(dict.fromkeys(int(i) for i in id_media_lista))
    if not ids:
        return []
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    segnaposto = ",".join("?" for _ in ids)
    cursore.execute(f"SELECT * FROM media_items WHERE id IN ({segnaposto})", ids)
    righe = cursore.fetchall()
    colonne = [col[0] for col in cursore.description]
    connessione.close()
    per_id = {r[0]: dict(zip(colonne, r)) for r in righe}
    return [per_id[i] for i in ids if i in per_id]

def conteggio_media_cercabili():
    """Numero di media con almeno un embedding, quindi realmente ricercabili."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("""
        SELECT COUNT(DISTINCT f.media_id)
        FROM media_frames f
        JOIN media_items m ON m.id = f.media_id
        WHERE m.processed = 1 AND f.clip_embedding IS NOT NULL
    """)
    conteggio = cursore.fetchone()[0]
    connessione.close()
    return conteggio

def ottieni_percorsi_archiviati():
    """Restituisce l'insieme (normalizzato) dei percorsi file registrati nel database.

    Serve a distinguere i file legittimi dell'archivio da quelli 'intrusi',
    cioè copiati manualmente in data/archive e mai indicizzati dall'app.
    """
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute("SELECT file_path FROM media_items")
    righe = cursore.fetchall()
    connessione.close()

    percorsi = set()
    for (percorso,) in righe:
        if percorso:
            percorsi.add(os.path.normcase(os.path.abspath(percorso)))
    return percorsi

def elimina_elemento_multimediale(id_media):
    """Elimina un elemento dal DB, i suoi frame, volti, vettori su ChromaDB e i relativi file fisici dal disco."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    
    # 1. Recupera gli ID per l'eliminazione dei vettori associati
    cursore.execute("SELECT id FROM media_frames WHERE media_id = ?", (id_media,))
    id_frame_lista = [r[0] for r in cursore.fetchall()]
    cursore.execute("SELECT id FROM faces WHERE media_id = ?", (id_media,))
    id_volto_lista = [r[0] for r in cursore.fetchall()]
    
    # 2. Raccoglie i percorsi dei file su disco per la pulizia successiva
    cursore.execute("SELECT image_path FROM media_frames WHERE media_id = ?", (id_media,))
    percorsi_frame = [r[0] for r in cursore.fetchall()]
    cursore.execute("SELECT crop_path FROM faces WHERE media_id = ?", (id_media,))
    percorsi_volti = [r[0] for r in cursore.fetchall()]
    cursore.execute("SELECT file_path FROM media_items WHERE id = ?", (id_media,))
    riga_originale = cursore.fetchone()
    percorso_originale = riga_originale[0] if riga_originale else None
    
    # 3. Elimina i record dal database (l'eliminazione a cascata cancella frame e volti correlati)
    cursore.execute("DELETE FROM media_items WHERE id = ?", (id_media,))
    connessione.commit()
    connessione.close()
    
    # 4. Rimuove i vettori dalle due collezioni Chroma
    if _store_frame is not None:
        try:
            _store_frame.elimina(id_frame_lista)
        except Exception as errore:
            print(f"Errore nella rimozione dei vettori frame per l'elemento {id_media}: {errore}")
    if _store_volti is not None:
        try:
            _store_volti.elimina(id_volto_lista)
        except Exception as errore:
            print(f"Errore nella rimozione dei vettori volto per l'elemento {id_media}: {errore}")
    
    # 5. Rimuove in sicurezza i file dal disco rigido
    for percorso in percorsi_frame + percorsi_volti:
        if percorso and os.path.exists(percorso):
            try:
                os.remove(percorso)
            except Exception as errore:
                print(f"Impossibile rimuovere il file {percorso}: {errore}")
                
    if percorso_originale:
        # Rimuove il file originale se ancora presente. Per le immagini potrebbe
        # essere gia' stato eliminato nel ciclo precedente (il frame di un'immagine
        # punta allo stesso file dell'archivio), quindi la rimozione della miniatura
        # NON deve dipendere dall'esistenza del file originale.
        if os.path.exists(percorso_originale):
            try:
                os.remove(percorso_originale)
            except Exception as errore:
                print(f"Errore durante la rimozione del file originale {percorso_originale}: {errore}")

        # Rimuove comunque l'eventuale miniatura associata
        try:
            percorso_anteprima = percorso_originale.replace(config.DIR_ARCHIVIO, config.DIR_ANTEPRIME)
            anteprima_base, _ = os.path.splitext(percorso_anteprima)
            for estensione in ['.jpg', '.png']:
                file_da_rimuovere = anteprima_base + estensione
                if os.path.exists(file_da_rimuovere):
                    os.remove(file_da_rimuovere)
        except Exception as errore:
            print(f"Errore durante la rimozione della miniatura per {percorso_originale}: {errore}")

def elimina_elementi_multimediali(id_media_lista):
    """Elimina più media, continuando sugli altri in caso di errore.

    Ritorna ``(id_eliminati, errori)``; ``errori`` mappa l'ID al messaggio.
    """
    eliminati = []
    errori = {}
    for id_media in dict.fromkeys(int(i) for i in id_media_lista):
        try:
            if ottieni_elemento_multimediale(id_media) is None:
                errori[id_media] = "Elemento non trovato"
                continue
            elimina_elemento_multimediale(id_media)
            if ottieni_elemento_multimediale(id_media) is None:
                eliminati.append(id_media)
            else:
                errori[id_media] = "Il record non è stato eliminato"
        except Exception as errore:
            errori[id_media] = str(errore)
    return eliminati, errori

def ottieni_statistiche_db():
    """Restituisce le statistiche sul database dell'archivio (conteggi, dimensioni, ecc.)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    
    cursore.execute("SELECT COUNT(*), SUM(file_size) FROM media_items")
    conteggio_totale, dimensione_totale = cursore.fetchone()
    dimensione_totale = dimensione_totale or 0
    
    cursore.execute("SELECT COUNT(*) FROM media_items WHERE media_type = 'image' AND processed = 1")
    conteggio_immagini = cursore.fetchone()[0]
    
    cursore.execute("SELECT COUNT(*) FROM media_items WHERE media_type = 'video' AND processed = 1")
    conteggio_video = cursore.fetchone()[0]
    
    cursore.execute("SELECT COUNT(*) FROM media_items WHERE processed = 0")
    conteggio_non_elaborati = cursore.fetchone()[0]
    
    cursore.execute("SELECT COUNT(*) FROM media_items WHERE processed = -1")
    conteggio_falliti = cursore.fetchone()[0]
    
    cursore.execute("SELECT COUNT(*) FROM media_frames")
    conteggio_frame = cursore.fetchone()[0]
    
    cursore.execute("SELECT COUNT(*) FROM faces")
    conteggio_volti = cursore.fetchone()[0]
    
    connessione.close()
    
    # Dimensione del file di database fisico
    dimensione_db = 0
    if os.path.exists(config.PERCORSO_DB):
        dimensione_db = os.path.getsize(config.PERCORSO_DB)
        
    return {
        "total_items": conteggio_totale,
        "total_size_bytes": dimensione_totale,
        "images_count": conteggio_immagini,
        "videos_count": conteggio_video,
        "unprocessed_count": conteggio_non_elaborati,
        "failed_count": conteggio_falliti,
        "frames_count": conteggio_frame,
        "faces_count": conteggio_volti,
        "db_size_bytes": dimensione_db
    }

# Colonne condivise per le query sui frame: lo stesso mapping riga->dizionario
# serve sia allo scan completo sia al recupero dei candidati da Chroma.
_SELECT_FRAME = """
    SELECT f.id, f.media_id, f.frame_index, f.timestamp_seconds, f.image_path, f.ocr_text, f.objects, f.clip_embedding,
           m.filename, m.file_path, m.media_type, m.creation_date, m.location_name
    FROM media_frames f
    JOIN media_items m ON f.media_id = m.id
    WHERE m.processed = 1
"""

def _riga_frame_a_dict(r):
    """Converte una riga di _SELECT_FRAME nel dizionario usato dalla UI."""
    emb = deserializza_vettore(r[7])
    if emb is None:
        return None
    lista_oggetti = []
    if r[6]:
        try:
            lista_oggetti = json.loads(r[6])
        except:
            lista_oggetti = [x.strip() for x in r[6].split(",") if x.strip()]
    return {
        "frame_id": r[0], "media_id": r[1], "frame_index": r[2],
        "timestamp_seconds": r[3], "image_path": r[4], "ocr_text": r[5] or "",
        "objects": lista_oggetti, "filename": r[8], "file_path": r[9],
        "media_type": r[10], "creation_date": r[11], "location_name": r[12] or "",
        "embedding": emb,
    }

def carica_tutti_embedding_clip():
    """Scan completo di tutti gli embedding CLIP dei frame (fallback se Chroma è assente)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute(_SELECT_FRAME)
    righe = cursore.fetchall()
    connessione.close()
    return [d for d in (_riga_frame_a_dict(r) for r in righe) if d is not None]

def cerca_frame_simili(query_emb, k=100):
    """Ricerca per similarità sui frame: Chroma restituisce i candidati, il coseno
    esatto viene ricalcolato in SQLite (punteggi e soglie identici allo scan lineare).

    Ritorna una lista di tuple (dizionario_frame, similarità_coseno).
    """
    try:
        if _store_frame is None:
            raise RuntimeError("indice vettoriale frame non disponibile")
        # ponytail: k=100 candidati; alzalo se i filtri della UI tagliano troppi risultati
        ids = _store_frame.cerca_simili(query_emb, migliori_k=k)
    except Exception as errore:
        # SQLite è la fonte di verità: se Chroma manca o l'indice HNSW fallisce
        # (es. accesso concorrente da più processi, corruzione) si ripiega sullo scan completo.
        if _store_frame is not None:
            print(f"Ricerca vettoriale frame fallita ({errore}); uso lo scan lineare SQLite.")
        return [(d, float(np.dot(query_emb, d["embedding"]))) for d in carica_tutti_embedding_clip()]

    if not ids:
        return []

    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    segnaposto = ",".join("?" * len(ids))
    cursore.execute(_SELECT_FRAME + f" AND f.id IN ({segnaposto})", ids)
    righe = cursore.fetchall()
    connessione.close()

    risultati = []
    for r in righe:
        d = _riga_frame_a_dict(r)
        if d is not None:
            risultati.append((d, float(np.dot(query_emb, d["embedding"]))))
    return risultati

_SELECT_VOLTO = """
    SELECT fc.id, fc.media_id, fc.frame_id, fc.crop_path, fc.embedding, fc.box_x1, fc.box_y1, fc.box_x2, fc.box_y2,
           m.filename, m.file_path, m.media_type, m.creation_date, m.location_name,
           fr.timestamp_seconds
    FROM faces fc
    JOIN media_items m ON fc.media_id = m.id
    LEFT JOIN media_frames fr ON fc.frame_id = fr.id
    WHERE m.processed = 1
"""

def _riga_volto_a_dict(r):
    """Converte una riga di _SELECT_VOLTO nel dizionario usato dalla UI."""
    emb = deserializza_vettore(r[4])
    if emb is None:
        return None
    return {
        "face_id": r[0], "media_id": r[1], "frame_id": r[2], "crop_path": r[3],
        "bbox": [r[5], r[6], r[7], r[8]], "filename": r[9], "file_path": r[10],
        "media_type": r[11], "creation_date": r[12], "location_name": r[13] or "",
        "timestamp_seconds": r[14] if r[14] is not None else 0.0, "embedding": emb,
    }

def carica_tutti_embedding_volti():
    """Scan completo di tutti gli embedding dei volti (fallback se Chroma è assente)."""
    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    cursore.execute(_SELECT_VOLTO)
    righe = cursore.fetchall()
    connessione.close()
    return [d for d in (_riga_volto_a_dict(r) for r in righe) if d is not None]

def cerca_volti_simili(query_emb, k=100):
    """Come cerca_frame_simili ma sui volti (embedding FaceNet)."""
    try:
        if _store_volti is None:
            raise RuntimeError("indice vettoriale volti non disponibile")
        ids = _store_volti.cerca_simili(query_emb, migliori_k=k)
    except Exception as errore:
        # Come per i frame: fallback allo scan lineare SQLite su qualsiasi errore Chroma.
        if _store_volti is not None:
            print(f"Ricerca vettoriale volti fallita ({errore}); uso lo scan lineare SQLite.")
        return [(d, float(np.dot(query_emb, d["embedding"]))) for d in carica_tutti_embedding_volti()]

    if not ids:
        return []

    connessione = ottieni_connessione()
    cursore = connessione.cursor()
    segnaposto = ",".join("?" * len(ids))
    cursore.execute(_SELECT_VOLTO + f" AND fc.id IN ({segnaposto})", ids)
    righe = cursore.fetchall()
    connessione.close()

    risultati = []
    for r in righe:
        d = _riga_volto_a_dict(r)
        if d is not None:
            risultati.append((d, float(np.dot(query_emb, d["embedding"]))))
    return risultati

def _sincronizza_indici_vettoriali():
    """Popola le collezioni Chroma dai BLOB in SQLite quando sono vuote.

    SQLite resta la fonte di verità; Chroma è solo l'indice di ricerca. Al primo
    avvio dopo aver separato frame e volti in due collezioni, le riempie dai dati
    già presenti così l'archivio esistente resta cercabile senza reimport.
    """
    try:
        if _store_frame is not None and _store_frame.conteggio() == 0:
            frames = carica_tutti_embedding_clip()
            if frames:
                _store_frame.aggiungi_o_aggiorna(
                    [d["frame_id"] for d in frames],
                    [d["embedding"] for d in frames],
                    [{"media_id": d["media_id"]} for d in frames],
                )
        if _store_volti is not None and _store_volti.conteggio() == 0:
            volti = carica_tutti_embedding_volti()
            if volti:
                _store_volti.aggiungi_o_aggiorna(
                    [d["face_id"] for d in volti],
                    [d["embedding"] for d in volti],
                    [{"media_id": d["media_id"]} for d in volti],
                )
    except Exception as errore:
        print(f"Sincronizzazione indici vettoriali fallita: {errore}")

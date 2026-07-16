import streamlit as st
import os
import sys
import subprocess
import datetime
import tempfile
import numpy as np
from PIL import Image, ImageOps

import config
import database
import processor
import gallery_utils
from models import gestore

# Percorso del logo blu (src/assets/logo_blu.png). Risolto da __file__ così vale
# indipendentemente dalla cartella di avvio del processo.
PERCORSO_LOGO = os.path.join(os.path.dirname(__file__), "assets", "logo_blu.png")

# Imposta la configurazione della pagina di Streamlit
st.set_page_config(
    page_title="DeepSight - Ricerca Foto e Video",
    page_icon=PERCORSO_LOGO,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializza il database locale all'avvio
database.inizializza_db()

# Backfill una tantum (offline): popola il nome del luogo dalle coordinate GPS per gli
# elementi già in archivio che ne sono privi. Eseguito una sola volta per sessione;
# se non c'è nulla da geocodificare esce subito senza caricare il dataset.
if "geocodifica_fatta" not in st.session_state:
    processor.geocodifica_luoghi_mancanti()
    st.session_state["geocodifica_fatta"] = True

# Avvia il lavoratore della coda: riprende da solo gli elementi pendenti nel DB.
processor.avvia_lavoratore()

# --- MARCHIO (in cima alla barra laterale) ---
# Logo blu grande e ben visibile. Il tema chiaro/scuro si cambia con lo switch NATIVO
# di Streamlit (icona ⋮ in alto a destra -> Settings -> Appearance). Per seguirlo senza
# doverlo rilevare (st.context.theme è inaffidabile all'istante del cambio, issue #11920),
# i colori del CSS sono NEUTRI/traslucidi: si adattano da soli sia al tema scuro sia al chiaro.
st.sidebar.image(PERCORSO_LOGO, width='stretch')
st.sidebar.markdown(
    "<h2 style='text-align:center; color:var(--accent-testo); margin-top:-6px;'>DeepSight</h2>",
    unsafe_allow_html=True,
)

# Stile CSS personalizzato per la UI. I colori sono NEUTRI (grigi traslucidi + accenti
# del brand), così restano leggibili su qualunque tema di Streamlit senza doverlo rilevare.
st.markdown("""
<style>
    /* Font Inter da Google Fonts (se offline, ricade sui sans-serif di sistema).
       L'@import DEVE precedere ogni altra regola: messo dopo, la spec CSS lo ignora. */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Token di design. Convenzione naming: i token nuovi sono in italiano
       (--gradiente-*, --raggio-*, --spazio-*, --accento-*); le varianti -rgb
       seguono la lingua del token base (--accent-1 -> --accent-1-rgb). */
    :root {
        --accent-1: #00C6FF;
        --accent-2: #0072FF;
        /* Canali RGB degli accenti: permettono rgba(var(--accent-N-rgb), alpha) senza
           ricopiare i valori numerici a ogni variante traslucida (aloni, ombre, focus). */
        --accent-1-rgb: 0, 198, 255;
        --accent-2-rgb: 0, 114, 255;
        --accent-testo: #1f8fd6;   /* blu medio leggibile sia su chiaro sia su scuro */
        --gradiente-accento: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        /* Stati semantici (avviso -> pericolo): usati dal pulsante scudo integrità e
           riusabili per futuri toast/messaggi d'errore. */
        --accento-avviso: #FF9800;
        --accento-pericolo: #F44336;
        --accento-pericolo-rgb: 244, 67, 54;
        --gradiente-pericolo: linear-gradient(135deg, var(--accento-avviso), var(--accento-pericolo));
        --superficie: rgba(130, 130, 150, 0.10);
        --superficie-2: rgba(130, 130, 150, 0.16);
        --bordo: rgba(130, 130, 150, 0.30);
        --ombra-card: rgba(0, 0, 0, 0.18);
        --scroll-thumb: rgba(130, 130, 150, 0.55);
        /* Scala dei raggi (valori esistenti, ora centralizzati) */
        --raggio-xs: 4px;
        --raggio-sm: 10px;
        --raggio-md: 12px;
        --raggio-lg: 16px;
        --raggio-pill: 20px;
        --raggio-nav: 24px;
        /* Scala delle spaziature */
        --spazio-1: 4px;
        --spazio-2: 8px;
        --spazio-3: 12px;
        --spazio-4: 16px;
        --spazio-5: 24px;
    }

    /* Applica Inter a tutta l'app */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Aloni blu del brand SOPRA lo sfondo del tema di Streamlit: si usa background-image
       (non 'background'), così il colore di sfondo del tema nativo scuro/chiaro resta. */
    .stApp {
        background-image:
            radial-gradient(1100px 550px at 12% -12%, rgba(var(--accent-2-rgb), 0.08), transparent 60%),
            radial-gradient(900px 500px at 105% -5%, rgba(var(--accent-1-rgb), 0.06), transparent 55%);
    }
    
    /* Nasconde l'icona-link ancora che Streamlit aggiunge accanto a ogni titolo */
    [data-testid="stHeaderActionElements"] { display: none; }

    /* Navbar sempre visibile: resta agganciata sotto l'header di Streamlit (alto 3.75rem)
       durante lo scroll. Lo sticky va sul wrapper di layout (non sul container keyed:
       il suo genitore è alto quanto lui, quindi lo sticky non avrebbe corsa).
       Sfondo traslucido + blur, così funziona su tema chiaro e scuro. */
    [data-testid="stLayoutWrapper"]:has(> .st-key-navbar) {
        position: sticky;
        top: 3.75rem;
        z-index: 99;
        background: var(--superficie);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 0.5rem 0.6rem;
        border-radius: var(--raggio-md);
    }

    /* Stile dei titoli */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: var(--gradiente-accento);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        opacity: 0.7;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Stile delle schede delle metriche */
    .metric-container {
        background: var(--superficie);
        border: 1px solid var(--bordo);
        border-radius: var(--raggio-lg);
        padding: var(--spazio-5);
        box-shadow: 0 4px 30px var(--ombra-card);
        backdrop-filter: blur(5px);
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: rgba(var(--accent-1-rgb), 0.3);
        box-shadow: 0 10px 30px rgba(var(--accent-1-rgb), 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-testo);
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Stile delle schede dei risultati */
    .result-card {
        background: var(--superficie);
        border: 1px solid var(--bordo);
        border-radius: var(--raggio-md);
        overflow: hidden;
        padding: 0;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 6px var(--ombra-card);
    }
    .result-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-2);
        box-shadow: 0 12px 24px rgba(var(--accent-2-rgb), 0.25);
    }
    .result-meta {
        padding: var(--spazio-3);
        font-size: 0.85rem;
    }
    .result-title {
        font-weight: 600;
        font-size: 0.95rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 6px;
    }
    .result-score {
        background: var(--gradiente-accento);
        color: white;
        padding: 2px 8px;
        border-radius: var(--raggio-pill);
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* Stile dei tag */
    .tag-pill {
        background: var(--superficie-2);
        padding: 2px var(--spazio-2);
        border-radius: var(--raggio-xs);
        font-size: 0.75rem;
        margin-right: var(--spazio-1);
        margin-bottom: var(--spazio-1);
        display: inline-block;
    }
    
    /* Stile pulsanti della barra di navigazione */
    div[data-testid="stColumn"] button {
        border-radius: var(--raggio-nav) !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        height: 48px !important;
        margin-bottom: 15px !important;
    }
    
    /* Pulsanti secondari (inattivi) */
    div[data-testid="stColumn"] button[kind="secondary"] {
        background: var(--superficie) !important;
        border: 1px solid var(--bordo) !important;
    }
    div[data-testid="stColumn"] button[kind="secondary"]:hover {
        color: var(--accent-testo) !important;
        border-color: var(--accent-1) !important;
        background: rgba(var(--accent-1-rgb), 0.05) !important;
        box-shadow: 0 0 15px rgba(var(--accent-1-rgb), 0.15) !important;
    }
    
    /* Pulsanti primari (attivi) */
    div[data-testid="stColumn"] button[kind="primary"] {
        background: var(--gradiente-accento) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(var(--accent-2-rgb), 0.35) !important;
    }
    div[data-testid="stColumn"] button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(var(--accent-2-rgb), 0.5) !important;
        transform: translateY(-1px);
    }

    /* Barra laterale: sfondo sfumato e bordo di separazione */
    [data-testid="stSidebar"] {
        background: var(--superficie);
        border-right: 1px solid var(--bordo);
    }

    /* Intestazioni di sezione (### ...) con barretta accento a sinistra */
    .stApp h3 {
        border-left: 3px solid var(--accent-1);
        padding-left: 10px;
        margin-top: 1.2rem;
    }

    /* Schede (tabs) più moderne */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid var(--bordo);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--raggio-sm) var(--raggio-sm) 0 0;
        padding: var(--spazio-2) var(--spazio-4);
        opacity: 0.7;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-testo) !important;
        opacity: 1;
        background: rgba(var(--accent-1-rgb), 0.06);
    }

    /* Campi di input, selectbox e date coerenti col tema */
    .stTextInput input,
    .stTextArea textarea,
    .stDateInput input,
    div[data-baseweb="select"] > div {
        background-color: var(--superficie) !important;
        border: 1px solid var(--bordo) !important;
        border-radius: var(--raggio-sm) !important;
    }
    .stTextInput input:focus,
    .stDateInput input:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 0 2px rgba(var(--accent-1-rgb), 0.15) !important;
    }

    /* Pulsanti generici (upload, azioni) arrotondati */
    .stButton > button {
        border-radius: var(--raggio-sm);
    }

    /* Riquadri espandibili coerenti col tema */
    [data-testid="stExpander"] {
        border: 1px solid var(--bordo);
        border-radius: var(--raggio-md);
        background: var(--superficie);
    }

    /* Griglia responsive: le tessere hanno dimensione stabile (~220px) su qualsiasi schermo e
       su finestre larghe ne entrano semplicemente di più per riga. Le colonne di Streamlit non
       lo permettono, essendo frazioni della larghezza: ogni elemento è quindi un container e il
       container con key diventa una CSS grid. */
    .st-key-dashboard_griglia,
    .st-key-galleria_griglia,
    .st-key-risultati_griglia {
        display: grid !important;
        /* Con 1fr le tracce si allargano a riempire la riga, quindi la tessera è sempre più
           larga del minimo, e di quanto dipende da dove cade la soglia della traccia in più.
           200px è il valore che rende la tessera più stabile (~205-215px) alle larghezze reali. */
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        align-items: start;
    }

    /* Griglie (galleria, dashboard, risultati di ricerca): card ad altezza
       uniforme, foto ritagliate in quadrati (object-fit: cover, stile Google
       Foto). L'originale resta intatto: è solo presentazione. */
    .st-key-galleria_griglia [data-testid="stImage"] img,
    .st-key-dashboard_griglia [data-testid="stImage"] img,
    .st-key-risultati_griglia [data-testid="stImage"] img {
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: var(--raggio-md);
    }
    /* Anche il player video inline della dashboard segue il quadrato
       (resta riproducibile: è ritagliato, non deformato) */
    .st-key-dashboard_griglia video[data-testid="stVideo"] {
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: var(--raggio-md);
    }
    /* A schermo intero il video resta figlio della griglia, quindi le regole di ritaglio
       qui sopra continuerebbero ad applicarsi: il filmato apparirebbe croppato e con gli
       angoli arrotondati. Vanno annullate esplicitamente. */
    .st-key-dashboard_griglia video[data-testid="stVideo"]:fullscreen {
        aspect-ratio: auto;
        object-fit: contain;
        border-radius: 0;
    }
    .st-key-dashboard_griglia video[data-testid="stVideo"]:-webkit-full-screen {
        aspect-ratio: auto;
        object-fit: contain;
        border-radius: 0;
    }

    /* Click sulla foto = apri a schermo intero, senza JavaScript.
       L'icona che Streamlit sovrappone alle immagini è un vero <button>: invece di
       nasconderla e simularne il click, la si allarga fino a coprire tutta la foto e la si
       rende invisibile. Cliccare l'immagine È cliccare il pulsante.
       Chiuso ed espanso si distinguono dall'aria-label del pulsante ("Fullscreen" /
       "Close fullscreen"): le classi st-emotion-cache-* sono hash instabili tra le versioni.
       I video non hanno toolbar, quindi non sono toccati. */
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stElementToolbar"] {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        padding: 0;
        opacity: 1;
        background: transparent !important;
    }
    /* Lo span del tooltip porta width:auto inline: senza !important resterebbe stretto. */
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stElementToolbar"] :is([data-testid="stElementToolbarButtonContainer"],
        [data-testid="stElementToolbarButton"], [data-testid="stTooltipHoverTarget"], button) {
        width: 100% !important;
        height: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        /* Il contenitore del pulsante ha uno sfondo opaco: allargato a tutta la foto la
           coprirebbe, perché è posizionato e dipinge sopra l'immagine. */
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stElementToolbar"] button {
        background: transparent !important;
        border: none !important;
        border-radius: var(--raggio-md);
        padding: 0 !important;
    }
    /* Stato chiuso: nessuna icona, cursore a lente d'ingrandimento. */
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stFullScreenFrame"]:has(button[aria-label="Fullscreen"]) button {
        cursor: zoom-in;
    }
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stFullScreenFrame"]:has(button[aria-label="Fullscreen"]) button svg {
        display: none;
    }
    /* Stato espanso: il clic ovunque chiude, ma la X resta visibile in alto a destra
       (con un'ombra per restare leggibile sopra la foto). Esc continua a funzionare. */
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"]) button {
        cursor: zoom-out;
        display: flex;
        align-items: flex-start;
        justify-content: flex-end;
        padding: var(--spazio-3) !important;
    }
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"]) button svg {
        filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.9));
    }
    /* Espansa, l'immagine resta figlia della griglia: le regole di ritaglio qui sopra
       continuerebbero ad applicarsi (angoli arrotondati). Vanno annullate. */
    :is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
    [data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"])
    [data-testid="stImage"] img {
        aspect-ratio: auto;
        object-fit: contain;
        border-radius: 0;
    }
    /* Il tooltip "Fullscreen" è renderizzato in un portale su <body>, quindi non lo si può
       nascondere per discendenza: lo si sopprime solo mentre il mouse è sopra una foto.
       Il tooltip del pulsante scudo (help=) resta intatto. */
    body:has(:is(.st-key-galleria_griglia, .st-key-dashboard_griglia, .st-key-risultati_griglia)
        [data-testid="stFullScreenFrame"]:hover) [data-testid="stTooltipContent"] {
        display: none !important;
    }
    /* Le immagini caricate per una ricerca (immagine di query, ritagli dei volti) sono
       immagini di lavoro, non elementi d'archivio: si toglie il pulsante fullscreen che
       Streamlit sovrappone a ogni st.image, così non sono ingrandibili. Le key dei crop nei
       dettagli sono dinamiche (indice) => match parziale. */
    :is(.st-key-immagine_query, .st-key-volti_query, [class*="st-key-crop_volto"])
    [data-testid="stElementToolbar"] {
        display: none !important;
    }
    /* Il blocco info delle card (data, luogo, dettagli — font 0.8rem) è fissato
       a due righe esatte: senza, i luoghi lunghi andando a capo sfalsano le
       righe della griglia. Il testo tagliato resta leggibile in "Azioni e
       Dettagli". */
    .st-key-galleria_griglia .result-card div[style*="0.8rem"],
    .st-key-dashboard_griglia .result-card div[style*="0.8rem"],
    .st-key-risultati_griglia .result-card div[style*="0.8rem"] {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 3.2em;
    }

    /* Pulsante flottante "+" per il caricamento (in basso a destra).
       Il pannello del popover è renderizzato in un portale fuori dal container,
       quindi questi stili toccano SOLO il pulsante circolare, non le opzioni. */
    .st-key-fab_caricamento {
        position: fixed;
        bottom: 28px;
        right: 28px;
        z-index: 1000;
        width: auto;
    }
    .st-key-fab_caricamento button {
        width: 60px !important;
        height: 60px !important;
        min-height: 60px !important;
        border-radius: 50% !important;
        font-size: 1.9rem !important;
        line-height: 1 !important;
        background: var(--gradiente-accento) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(var(--accent-2-rgb), 0.45) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .st-key-fab_caricamento button:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 10px 28px rgba(var(--accent-2-rgb), 0.6) !important;
    }
    /* Nasconde la freccetta del popover accanto al "+". In Streamlit 1.59 non è un <svg>
       ma un'icona material testuale ("expand_more"), da cui il selettore sul data-testid.
       Va nascosto l'intero contenitore e non la sola icona: il wrapper vuoto continuerebbe
       a occupare larghezza e gap nel flex, spostando il "+" fuori dal centro del cerchio. */
    .st-key-fab_caricamento button div:has(> span > [data-testid="stIconMaterial"]) { display: none; }
    .st-key-fab_caricamento button { padding: 0 !important; }
    /* Streamlit compensa lo spazio della freccetta con un margin-right negativo: azzerarlo
       è necessario, altrimenti il "+" resta scentrato di quei pixel. */
    .st-key-fab_caricamento button > div,
    .st-key-nav_integrita button > div {
        gap: 0 !important;
        margin: 0 !important;
        justify-content: center !important;
        width: 100%;
    }
    /* Il font-size del bottone non arriva al <p> del markdown, che ha il suo: senza questa
       regola il "+" resterebbe a 14px dentro un cerchio da 60. */
    .st-key-fab_caricamento button p {
        font-size: 1.9rem !important;
        line-height: 1 !important;
    }

    /* Pulsante scudo del controllo integrità, nella navbar. Come per il "+", il pannello
       a comparsa vive in un portale fuori dal container: questi stili toccano solo il bottone. */
    .st-key-nav_integrita button {
        height: 100%;
        border-radius: var(--raggio-sm) !important;
    }
    .st-key-nav_integrita button div:has(> span > [data-testid="stIconMaterial"]) { display: none; }
    /* Con problemi rilevati il pulsante diventa 'primary': accento ambra→rosso per farsi notare.
       Il doppio attributo serve a superare la specificità di
       `div[data-testid="stColumn"] button[kind="primary"]`, che colora di blu i pulsanti attivi. */
    .st-key-nav_integrita button[data-testid="stPopoverButton"][kind="primary"] {
        background: var(--gradiente-pericolo) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(var(--accento-pericolo-rgb), 0.35) !important;
    }
    .st-key-nav_integrita button[data-testid="stPopoverButton"][kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(var(--accento-pericolo-rgb), 0.5) !important;
    }

    /* Barra di scorrimento personalizzata */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--scroll-thumb);
        border-radius: 6px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-2); }
</style>
""", unsafe_allow_html=True)

def percorso_anteprima_elemento(percorso_file):
    """Percorso della miniatura di un elemento archiviato, o None se non esiste.

    Le miniature sono salvate con il nome-hash del file archiviato (basename di
    file_path), NON con il filename originale: la lookup va fatta da file_path.
    """
    nome_senza_est, _ = os.path.splitext(os.path.basename(percorso_file))
    percorso = os.path.join(config.DIR_ANTEPRIME, f"{nome_senza_est}.jpg")
    return percorso if os.path.exists(percorso) else None

def immagine_per_display(percorso, lato_max=None):
    """Apre un'immagine applicando l'orientamento EXIF, per passarla a st.image.

    Passare a st.image il PERCORSO di una foto più larga della pagina fa sì che
    Streamlit la ricodifichi con PIL perdendo il tag EXIF Orientation senza
    applicarlo: le foto verticali da telefono apparirebbero orizzontali.

    Con `lato_max` la foto viene ridotta a quel lato massimo: per le griglie si usa
    la foto originale (nitida, a differenza delle miniature da 300px stirate) ma
    ridimensionata. La decodifica JPEG avviene già a scala ridotta (draft), quindi
    il costo è vicino a quello di una miniatura.
    """
    try:
        with Image.open(percorso) as img:
            if lato_max:
                img.draft("RGB", (lato_max, lato_max))
            img = ImageOps.exif_transpose(img)
            if lato_max:
                img.thumbnail((lato_max, lato_max))
            img.load()
        return img
    except Exception:
        # Se PIL non riesce ad aprirla, lascia fare a st.image
        return percorso

def mostra_player_video(percorso, **kwargs):
    """Player per gli elementi 'video'. Le GIF sono video per la pipeline ma
    st.video non le riproduce (HTML5 <video> non supporta image/gif): st.image
    invece le anima nativamente nel browser. kwargs (es. start_time) valgono
    solo per i veri video."""
    if percorso.lower().endswith(".gif"):
        st.image(percorso, width="stretch")
    else:
        st.video(percorso, **kwargs)

def ottieni_stringa_dimensione_file(dimensione_in_byte):
    """Converte una dimensione in byte in formato leggibile (KB, MB, GB)."""
    for unita in ['B', 'KB', 'MB', 'GB']:
        if dimensione_in_byte < 1024.0:
            return f"{dimensione_in_byte:.2f} {unita}"
        dimensione_in_byte /= 1024.0
    return f"{dimensione_in_byte:.2f} TB"

def apri_cartella(percorso):
    """Apre una cartella nel gestore file del sistema. Ritorna True se riuscito.

    Ha senso solo perché DeepSight è un'app locale: il processo Streamlit gira
    sulla stessa macchina del browser, quindi la finestra si apre davanti all'utente.
    """
    try:
        os.makedirs(percorso, exist_ok=True)
        if os.name == "nt":
            os.startfile(percorso)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", percorso])
        else:
            subprocess.Popen(["xdg-open", percorso])
        return True
    except Exception:
        return False

def elenca_file_quarantena():
    """Nomi dei file attualmente isolati in data/quarantena."""
    try:
        return [n for n in os.listdir(config.DIR_QUARANTENA)
                if os.path.isfile(os.path.join(config.DIR_QUARANTENA, n))]
    except OSError:
        return []

def pannello_integrita(file_intrusi, record_orfani):
    """Contenuto del popover 🛡️: file 'intrusi' nell'archivio e record orfani nel database.

    Le due liste usano contenitori scrollabili invece di expander: dentro un popover un
    expander che si apre farebbe saltare l'altezza del pannello fluttuante.
    """
    st.caption(
        "La cartella `archive` è gestita dall'app: i file copiati a mano non vengono "
        "indicizzati (niente tag, volti, testo o ricerca), mentre quelli rimossi a mano "
        "lasciano nel database record 'orfani'. Qui puoi rilevare entrambi i casi e sistemarli."
    )

    if not file_intrusi:
        st.success("Nessun file intruso rilevato: l'archivio è allineato con il database.")
    else:
        st.warning(f"Rilevati **{len(file_intrusi)}** file non indicizzati (aggiunti manualmente).")
        # Altezza fissa (quindi scrollabile) solo se la lista è lunga, altrimenti il riquadro
        # resterebbe mezzo vuoto.
        with st.container(height=160 if len(file_intrusi) > 5 else "content", border=True):
            for percorso in file_intrusi:
                try:
                    dimensione = ottieni_stringa_dimensione_file(os.path.getsize(percorso))
                except OSError:
                    dimensione = "N/D"
                st.write(f"- `{os.path.basename(percorso)}` — {dimensione}")

        if st.button("📥 Importa e indicizza", key="btn_importa_intrusi", width='stretch'):
            with st.spinner("Importazione e indicizzazione in corso..."):
                importati, in_quarantena, falliti = processor.importa_file_intrusi()
            st.session_state.esito_integrita = {
                "icona": "✅",
                "testo": (
                    f"Importati e indicizzati: {importati} · "
                    f"Spostati in quarantena (formato non supportato): {in_quarantena} · "
                    f"Falliti: {falliti}."
                ),
            }
            st.rerun()
        if st.button("🧹 Sposta in quarantena", key="btn_quarantena_intrusi", width='stretch'):
            spostati = processor.sposta_file_intrusi_in_quarantena()
            st.session_state.esito_integrita = {
                "icona": "🧹",
                "testo": f"Spostati in quarantena: {spostati} file. Restano in data/quarantena, non indicizzati.",
            }
            st.rerun()

    # Il pulsante dipende dal contenuto della cartella, non dalla notifica: così Streamlit
    # lo ricrea a ogni run e il click resta registrabile anche dopo che la notifica sparisce.
    file_quarantena = elenca_file_quarantena()
    if file_quarantena:
        if st.button(f"📂 Apri cartella quarantena ({len(file_quarantena)})",
                     key="btn_apri_quarantena", width='stretch'):
            if not apri_cartella(config.DIR_QUARANTENA):
                st.warning(f"Impossibile aprire la cartella: `{config.DIR_QUARANTENA}`")

    # Record orfani: registrati nel database ma senza più il file su disco
    if not record_orfani:
        st.success("Nessun record orfano: tutti i file registrati esistono su disco.")
    else:
        st.warning(
            f"Rilevati **{len(record_orfani)}** record orfani: file registrati nel database "
            "ma rimossi dal disco. Compaiono ancora in ricerca e galleria (solo miniatura)."
        )
        with st.container(height=160 if len(record_orfani) > 5 else "content", border=True):
            etichette_stato = {1: "elaborato", 0: "in coda", -1: "fallito"}
            for orfano in record_orfani:
                stato = etichette_stato.get(orfano["processed"], "sconosciuto")
                st.write(f"- `{orfano['filename']}` — {orfano['media_type']}, {stato}")
        if st.button("🧹 Rimuovi dall'indice", key="btn_rimuovi_orfani", width='stretch'):
            rimossi = processor.rimuovi_record_orfani()
            st.session_state.esito_integrita = {
                "icona": "✅",
                "testo": (
                    f"Rimossi {rimossi} record orfani, con relativi vettori, miniature, "
                    "frame e volti associati."
                ),
            }
            st.rerun()

def applica_filtri(elemento, dizionario_filtri):
    """Verifica se un elemento rispetta i criteri di filtro definiti."""
    if not dizionario_filtri:
        return True
        
    if "media_type" in dizionario_filtri and dizionario_filtri["media_type"] != "Tutti":
        tipo_filtro = "image" if dizionario_filtri["media_type"] == "Immagini" else "video"
        if elemento.get("media_type") != tipo_filtro:
            return False
            
    # Come per il filtro luogo: con il filtro data attivo, gli elementi senza data
    # (o con data non interpretabile) vengono esclusi, non inclusi per default.
    if dizionario_filtri.get("date_start") or dizionario_filtri.get("date_end"):
        try:
            data_elemento = datetime.date.fromisoformat(elemento["creation_date"].split("T")[0])
        except (KeyError, TypeError, AttributeError, ValueError):
            return False
        if dizionario_filtri.get("date_start") and data_elemento < dizionario_filtri["date_start"]:
            return False
        if dizionario_filtri.get("date_end") and data_elemento > dizionario_filtri["date_end"]:
            return False
            
    if dizionario_filtri.get("location"):
        # Confronta col nome del luogo dell'elemento. Se l'elemento non ha un luogo
        # (stringa vuota) NON deve passare il filtro: quando si filtra per luogo, gli
        # elementi senza luogo vanno esclusi (prima venivano erroneamente inclusi).
        luogo_elemento = (elemento.get("location_name") or "").lower()
        if dizionario_filtri["location"].lower() not in luogo_elemento:
            return False

    return True

def deduplica_risultati(risultati_ricerca):
    """Mantiene solo il risultato con punteggio migliore per ciascun media_id."""
    migliore_per_media = {}
    for elemento, punteggio, modalita in risultati_ricerca:
        id_media = elemento.get("media_id")
        if id_media not in migliore_per_media or punteggio > migliore_per_media[id_media][1]:
            migliore_per_media[id_media] = (elemento, punteggio, modalita)
    risultati_deduplicati = list(migliore_per_media.values())
    risultati_deduplicati.sort(key=lambda x: x[1], reverse=True)
    return risultati_deduplicati


CHIAVE_SELEZIONE_GALLERIA = "galleria_id_selezionati"
PREFISSO_CHECKBOX_GALLERIA = "gal_sel_"


def id_selezionati_galleria():
    return set(st.session_state.get(CHIAVE_SELEZIONE_GALLERIA, []))


def invalida_zip_galleria():
    percorso = st.session_state.pop("galleria_zip_path", None)
    st.session_state.pop("galleria_zip_ids", None)
    st.session_state.pop("galleria_zip_mancanti", None)
    if percorso and os.path.exists(percorso):
        try:
            os.remove(percorso)
        except OSError:
            pass


def salva_selezione_galleria(ids, aggiorna_checkbox=False):
    ids = sorted(set(int(i) for i in ids))
    if ids != st.session_state.get(CHIAVE_SELEZIONE_GALLERIA, []):
        invalida_zip_galleria()
    st.session_state[CHIAVE_SELEZIONE_GALLERIA] = ids
    if aggiorna_checkbox:
        insieme = set(ids)
        for chiave in list(st.session_state):
            if chiave.startswith(PREFISSO_CHECKBOX_GALLERIA):
                try:
                    id_media = int(chiave[len(PREFISSO_CHECKBOX_GALLERIA):])
                    st.session_state[chiave] = id_media in insieme
                except ValueError:
                    continue


def aggiorna_selezione_da_checkbox(id_media):
    selezionati = id_selezionati_galleria()
    if st.session_state.get(f"{PREFISSO_CHECKBOX_GALLERIA}{id_media}"):
        selezionati.add(id_media)
    else:
        selezionati.discard(id_media)
    salva_selezione_galleria(selezionati)


def richiedi_conferma_eliminazione(ids):
    st.session_state["eliminazione_da_confermare"] = sorted(set(int(i) for i in ids))
    st.rerun()


@st.dialog("Conferma eliminazione")
def dialogo_conferma_eliminazione():
    ids = st.session_state.get("eliminazione_da_confermare", [])
    elementi = database.ottieni_elementi_multimediali(ids)
    if not elementi:
        st.warning("Gli elementi selezionati non sono più presenti nell'archivio.")
        if st.button("Chiudi", width="stretch"):
            st.session_state.pop("eliminazione_da_confermare", None)
            st.rerun()
        return

    quanti = len(elementi)
    st.warning(
        f"Stai per eliminare definitivamente **{quanti} "
        f"{'elemento' if quanti == 1 else 'elementi'}**. Verranno rimossi anche "
        "gli originali e tutti i dati associati. Vuoi continuare?"
    )
    with st.container(height=180 if quanti > 6 else "content", border=True):
        for elemento in elementi:
            st.write(f"- `{elemento['filename']}`")

    col_annulla, col_elimina = st.columns(2)
    if col_annulla.button("Annulla", width="stretch"):
        st.session_state.pop("eliminazione_da_confermare", None)
        st.rerun()
    if col_elimina.button(
        f"Elimina {quanti} {'elemento' if quanti == 1 else 'elementi'}",
        type="primary",
        width="stretch",
    ):
        eliminati, errori = database.elimina_elementi_multimediali(ids)
        selezionati = id_selezionati_galleria() - set(eliminati)
        salva_selezione_galleria(selezionati, aggiorna_checkbox=True)
        st.session_state.pop("eliminazione_da_confermare", None)
        st.session_state["esito_eliminazione"] = {
            "eliminati": len(eliminati),
            "errori": len(errori),
        }
        st.rerun()


def prepara_zip_galleria(ids):
    invalida_zip_galleria()
    elementi = database.ottieni_elementi_multimediali(ids)
    temporaneo = tempfile.NamedTemporaryFile(prefix="deepsight-galleria-", suffix=".zip", delete=False)
    percorso = temporaneo.name
    temporaneo.close()
    try:
        inclusi, mancanti = gallery_utils.crea_zip_originali(elementi, percorso)
    except Exception:
        if os.path.exists(percorso):
            os.remove(percorso)
        raise
    if not inclusi:
        if os.path.exists(percorso):
            os.remove(percorso)
        return [], mancanti
    st.session_state["galleria_zip_path"] = percorso
    st.session_state["galleria_zip_ids"] = sorted(ids)
    st.session_state["galleria_zip_mancanti"] = mancanti
    return inclusi, mancanti


def vai_a_pagina_galleria(pagina):
    st.session_state["gal_pagina_corrente"] = int(pagina)
    st.rerun()


def mostra_paginatore_galleria(
    pagina_corrente, totale_pagine, totale_elementi, inizio, fine, scorrimento_continuo=False
):
    """Controlli esclusivamente in fondo alla Galleria, senza tendina della pagina."""
    st.markdown("---")
    if scorrimento_continuo:
        st.caption(f"{totale_elementi} elementi · scorrimento continuo")
    else:
        st.caption(
            f"Elementi {inizio + 1 if totale_elementi else 0}–{fine} di {totale_elementi} "
            f"· pagina {pagina_corrente} di {totale_pagine}"
        )
    col_quantita, col_spazio = st.columns([1, 3])
    with col_quantita:
        st.selectbox(
            "Visualizzazione",
            ["16", "24", "48", "96", "continuo"],
            key="gal_elementi_per_pagina",
            format_func=lambda valore: (
                "Scorrimento continuo" if valore == "continuo" else f"{valore} per pagina"
            ),
        )

    if scorrimento_continuo:
        return

    pagine = gallery_utils.pagine_compatte(pagina_corrente, totale_pagine)
    larghezze = [1.35] + [0.55] * len(pagine) + [1.35]
    colonne = st.columns(larghezze)
    if colonne[0].button(
        "← Precedente", key="gal_pag_precedente", width="stretch",
        disabled=pagina_corrente <= 1,
    ):
        vai_a_pagina_galleria(pagina_corrente - 1)

    for indice, pagina in enumerate(pagine, start=1):
        if pagina is None:
            colonne[indice].markdown(
                "<div style='text-align:center;padding-top:0.45rem'>…</div>",
                unsafe_allow_html=True,
            )
            continue
        if colonne[indice].button(
            str(pagina),
            key=f"gal_pag_num_{pagina}",
            type="primary" if pagina == pagina_corrente else "secondary",
            disabled=pagina == pagina_corrente,
            width="stretch",
        ):
            vai_a_pagina_galleria(pagina)

    if colonne[-1].button(
        "Successiva →", key="gal_pag_successiva", width="stretch",
        disabled=pagina_corrente >= totale_pagine,
    ):
        vai_a_pagina_galleria(pagina_corrente + 1)


# (I filtri di ricerca sono stati spostati nella pagina "Ricerca Avanzata".)

# Pannello coda: si auto-aggiorna ogni secondo ed e' visibile da ogni pagina.
@st.fragment(run_every=1.0)
def pannello_coda():
    stato = processor.stato_coda()
    if stato["rimanenti"] == 0 and not stato["in_corso"]:
        # Anche a coda vuota il pulsante di retry deve esserci: se TUTTI gli stadi
        # falliscono (es. modelli Qwen mancanti al primo avvio) la coda si svuota
        # e senza questo pulsante i falliti resterebbero irrecuperabili dalla UI.
        if stato["falliti"]:
            st.warning(f"⚠️ {stato['falliti']} elementi con elaborazioni fallite")
            if st.button(f"🔁 Riprova falliti ({stato['falliti']})", key="coda_riprova_vuota", width='stretch'):
                processor.riprova_falliti()
                st.rerun(scope="fragment")
        return
    riga = f"⚙️ **Coda elaborazione:** {stato['rimanenti']} file"
    if stato["in_corso"]:
        riga += f"\n\n`{stato['in_corso']}`"
    if stato["eta_secondi"]:
        riga += f"\n\n⏳ stimati {datetime.timedelta(seconds=int(stato['eta_secondi']))}"
    st.info(riga)
    st.caption(f"da preparare: {stato['da_preparare']} · embedding: {stato['embedding']} · "
               f"volti: {stato['volti']} · trascrizioni: {stato['trascrizione']}")
    if stato["in_pausa"]:
        st.warning("⏸️ Coda in pausa (l'elemento corrente viene completato)")
        if st.button("▶️ Riprendi", key="coda_riprendi", width='stretch'):
            processor.riprendi()
            st.rerun(scope="fragment")
    else:
        if st.button("⏸️ Metti in pausa", key="coda_pausa", width='stretch'):
            processor.metti_in_pausa()
            st.rerun(scope="fragment")
    if stato["falliti"]:
        if st.button(f"🔁 Riprova falliti ({stato['falliti']})", key="coda_riprova", width='stretch'):
            processor.riprova_falliti()
            st.rerun(scope="fragment")

with st.sidebar:
    pannello_coda()

# Informazioni locali e pulizia memoria
st.sidebar.markdown("### Configurazione Locale")
st.sidebar.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Dispositivo:** `{gestore.dispositivo.upper()}`")
st.sidebar.markdown(f"**Intervallo Frame:** `{config.INTERVALLO_FRAME_VIDEO} sec`")
st.sidebar.markdown(f"**Soglia Volti:** `{config.SOGLIA_SIMILARITA_VOLTI}`")

if gestore.dispositivo == "cuda" and st.sidebar.button("Libera Memoria GPU", key="sb_clear_gpu"):
    gestore.libera_memoria()
    st.sidebar.success("Memoria liberata con successo!")

# --- NAVIGAZIONE SUPERIORE (NAVBAR) ---
if "selezione_menu" not in st.session_state:
    st.session_state.selezione_menu = "📊 Dashboard"

# Scansioni di integrità: servono sia al badge dello scudo sia al pannello del popover,
# quindi si calcolano una volta sola per rerun e si passano al pannello.
file_intrusi = processor.trova_file_intrusi()
record_orfani = processor.trova_record_orfani()
n_problemi = len(file_intrusi) + len(record_orfani)

# Esito dell'ultima azione, emesso qui e non dentro il popover: st.rerun() scarta i messaggi
# del run in cui l'azione è avvenuta, e al run successivo il pannello può essere già chiuso.
esito = st.session_state.pop("esito_integrita", None)
if esito:
    st.toast(esito["testo"], icon=esito["icona"])
esito_eliminazione = st.session_state.pop("esito_eliminazione", None)
if esito_eliminazione:
    eliminati = esito_eliminazione["eliminati"]
    errori = esito_eliminazione["errori"]
    if eliminati:
        st.toast(f"Eliminati {eliminati} elementi dall'archivio.", icon="✅")
    if errori:
        st.toast(f"Impossibile eliminare {errori} elementi.", icon="⚠️")

contenitore_navbar = st.container(key="navbar")
with contenitore_navbar:
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([1, 1, 1, 1, 0.35])

# (etichetta, key, colonna): l'etichetta è anche il valore salvato in selezione_menu.
voci_navbar = [
    ("📊 Dashboard", "nav_dash", col_nav1),
    ("🖼️ Galleria", "nav_gallery", col_nav2),
    ("🔍 Ricerca Avanzata", "nav_search", col_nav3),
    ("👤 Persone", "nav_persone", col_nav4),
]
for etichetta, key, colonna in voci_navbar:
    attivo = st.session_state.selezione_menu == etichetta
    if colonna.button(etichetta, width='stretch', key=key, type="primary" if attivo else "secondary"):
        st.session_state.selezione_menu = etichetta
        st.rerun()

# Scudo: apre il controllo integrità in un pannello a comparsa. Il badge mostra quanti
# problemi sono stati rilevati (file intrusi + record orfani).
with col_nav5:
    with st.container(key="nav_integrita"):
        etichetta_scudo = f"🛡️ {n_problemi}" if n_problemi else "🛡️"
        with st.popover(etichetta_scudo, width='stretch',
                        help="Controllo integrità archivio",
                        type="primary" if n_problemi else "secondary"):
            pannello_integrita(file_intrusi, record_orfani)

# --- PULSANTE FLOTTANTE "+" (in basso a destra, visibile su ogni pagina) ---
# Apre le opzioni di caricamento: la pagina Caricamento non è più nella navbar.
with st.container(key="fab_caricamento"):
    with st.popover("➕"):
        st.markdown("**Aggiungi contenuti**")
        if st.button("📤 Carica file", key="fab_upload_file", width='stretch'):
            st.session_state.selezione_menu = "📤 Caricamento & Import"
            st.session_state.modalita_caricamento = "file"
            st.rerun()
        if st.button("📂 Scansiona cartella", key="fab_scan_cartella", width='stretch'):
            st.session_state.selezione_menu = "📤 Caricamento & Import"
            st.session_state.modalita_caricamento = "cartella"
            st.rerun()

st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 2rem; border-color: var(--bordo);'>", unsafe_allow_html=True)

menu = st.session_state.selezione_menu

if st.session_state.get("eliminazione_da_confermare"):
    dialogo_conferma_eliminazione()


# --- 1. SCHEDA DASHBOARD ---
if menu == "📊 Dashboard":
    st.markdown("<h1 class='main-title'>Dashboard Archivio</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Statistiche dell'archivio privato multimediale locale</p>", unsafe_allow_html=True)
    
    statistiche = database.ottieni_statistiche_db()
    
    # 4 Colonne per le metriche principali
    metriche = [
        (statistiche['images_count'] + statistiche['videos_count'], "Elementi Indicizzati"),
        (statistiche['frames_count'], "Frame Estratti"),
        (statistiche['faces_count'], "Volti Rilevati"),
        (ottieni_stringa_dimensione_file(statistiche['total_size_bytes']), "Dimensione Archivio"),
    ]
    for colonna, (valore, etichetta) in zip(st.columns(4), metriche):
        colonna.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{valore}</div>
            <div class="metric-label">{etichetta}</div>
        </div>
        """, unsafe_allow_html=True)

    # Statistiche dettagliate
    st.markdown("### Dettaglio Risorse")
    col_dettaglio1, col_dettaglio2 = st.columns(2)
    
    with col_dettaglio1:
        st.write(f"- **Immagini elaborate:** {statistiche['images_count']}")
        st.write(f"- **Video elaborati:** {statistiche['videos_count']}")
        st.write(f"- **File da preparare:** {statistiche['unprocessed_count']}")
        st.write(f"- **File falliti:** {statistiche['failed_count']}")
    
    with col_dettaglio2:
        st.write(f"- **Dimensione Database SQLite:** {ottieni_stringa_dimensione_file(statistiche['db_size_bytes'])}")
        st.write(f"- **Cartella Archivio:** `{config.DIR_ARCHIVIO}`")
        st.write(f"- **Cartella Database:** `{config.DIR_DB}`")

    # Elementi aggiunti di recente
    st.markdown("### Elementi Aggiunti di Recente")
    elementi_recenti = database.ottieni_tutti_elementi_multimediali(solo_elaborati=True)
    if elementi_recenti:
        # Ordina per ID decrescente per ottenere i più recenti
        elementi_recenti.sort(key=lambda x: x["id"], reverse=True)
        elementi_recenti = elementi_recenti[:12] # Primi 12

        # Container con key: il CSS lo trasforma in una griglia responsive e ritaglia le foto
        # in quadrati uniformi. Un container per elemento = una cella della griglia.
        with st.container(key="dashboard_griglia"):
            for elemento in elementi_recenti:
                with st.container():
                    # Per le immagini si mostra la foto vera ridotta al volo (nitida);
                    # la miniatura da 300px resta solo come ripiego se l'originale manca.
                    nome_file = elemento["filename"]
                    if elemento["media_type"] == "image" and os.path.exists(elemento["file_path"]):
                        percorso_anteprima = elemento["file_path"]
                    else:
                        percorso_anteprima = percorso_anteprima_elemento(elemento["file_path"])

                    # Card HTML
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-meta">
                            <div class="result-title">{nome_file}</div>
                            <div style="opacity:0.7; font-size:0.8rem;">
                                Tipo: {elemento['media_type'].upper()} | {elemento['width']}x{elemento['height']} <br>
                                📅 {elemento['creation_date'].split('T')[0] if elemento['creation_date'] else 'N/D'} &nbsp;·&nbsp; 📍 {elemento.get('location_name') or 'N/D'}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Per i video mostra un vero player (riproducibile); per le immagini
                    # la miniatura. Se il video non fosse disponibile si ripiega sulla miniatura.
                    if elemento["media_type"] == "video" and os.path.exists(elemento["file_path"]):
                        mostra_player_video(elemento["file_path"])
                    elif percorso_anteprima and os.path.exists(percorso_anteprima):
                        st.image(immagine_per_display(percorso_anteprima, lato_max=1000), width="stretch")
    else:
        st.info("Nessun elemento presente nell'archivio. Vai alla scheda 'Caricamento' per importare contenuti.")


# --- 2. SCHEDA GALLERIA ---
elif menu == "🖼️ Galleria":
    st.markdown("<h1 class='main-title'>Galleria</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Sfoglia tutti gli elementi dell'archivio, senza bisogno di una ricerca</p>", unsafe_allow_html=True)

    elementi_galleria = database.ottieni_tutti_elementi_multimediali(solo_elaborati=True)
    if not elementi_galleria:
        st.info("Nessun elemento presente nell'archivio. Vai alla scheda 'Caricamento' per importare contenuti.")
    else:
        # In cima restano solo ricerca e filtri. Tutta la paginazione è in fondo.
        col_gal1, col_gal2, col_gal3 = st.columns([2, 1, 1])
        with col_gal1:
            ricerca_galleria = st.text_input(
                "Cerca nella galleria",
                placeholder="Nome, luogo o data (es. 16/07/2026)",
                key="gal_ricerca_semplice",
            )
        with col_gal2:
            tipo_galleria = st.selectbox("Tipo di file", ["Tutti", "Immagini", "Video"], key="gal_tipo")
        with col_gal3:
            ordinamento_galleria = st.selectbox("Ordina per", ["Più recenti", "Meno recenti", "Nome (A-Z)"], key="gal_ordine")

        elementi_filtrati = gallery_utils.filtra_e_ordina_elementi(
            elementi_galleria,
            ricerca_galleria,
            tipo_galleria,
            ordinamento_galleria,
        )

        # Mantiene solo ID ancora esistenti, senza perdere la selezione tra le pagine.
        ids_esistenti = {e["id"] for e in elementi_galleria}
        selezionati = id_selezionati_galleria() & ids_esistenti
        salva_selezione_galleria(selezionati)

        if "gal_elementi_per_pagina" not in st.session_state:
            st.session_state["gal_elementi_per_pagina"] = "24"
        elif isinstance(st.session_state["gal_elementi_per_pagina"], int):
            # Compatibilità con sessioni avviate prima dell'opzione di scorrimento continuo.
            st.session_state["gal_elementi_per_pagina"] = str(
                st.session_state["gal_elementi_per_pagina"]
            )
        visualizzazione = st.session_state["gal_elementi_per_pagina"]
        scorrimento_continuo = visualizzazione == "continuo"
        elementi_per_pagina = None if scorrimento_continuo else int(visualizzazione)

        firma_filtri = (
            ricerca_galleria.casefold().strip(),
            tipo_galleria,
            ordinamento_galleria,
            visualizzazione,
        )
        if st.session_state.get("gal_firma_filtri") != firma_filtri:
            st.session_state["gal_firma_filtri"] = firma_filtri
            st.session_state["gal_pagina_corrente"] = 1

        totale_elementi = len(elementi_filtrati)
        totale_pagine = (
            1 if scorrimento_continuo
            else max(1, -(-totale_elementi // elementi_per_pagina))
        )
        pagina_corrente = min(
            max(1, st.session_state.get("gal_pagina_corrente", 1)),
            totale_pagine,
        )
        st.session_state["gal_pagina_corrente"] = pagina_corrente
        inizio_pagina = 0 if scorrimento_continuo else (pagina_corrente - 1) * elementi_per_pagina
        fine_pagina = (
            totale_elementi if scorrimento_continuo
            else min(inizio_pagina + elementi_per_pagina, totale_elementi)
        )
        elementi_visibili = elementi_filtrati[inizio_pagina:fine_pagina]
        ids_visibili = {e["id"] for e in elementi_visibili}

        # Azioni di selezione, separate dalla paginazione.
        n_visibili_selezionati = len(selezionati & ids_visibili)
        n_nascosti_selezionati = len(selezionati - ids_visibili)
        testo_selezione = f"**{len(selezionati)} selezionati** · {n_visibili_selezionati} in questa pagina"
        if n_nascosti_selezionati:
            testo_selezione += f" · {n_nascosti_selezionati} non visibili"
        st.markdown(testo_selezione)

        col_sel1, col_sel2, col_sel3, col_zip, col_del = st.columns([1, 1, 1, 1, 1])
        if col_sel1.button("Seleziona pagina", width="stretch", disabled=not elementi_visibili):
            salva_selezione_galleria(selezionati | ids_visibili, aggiorna_checkbox=True)
            st.rerun()
        if col_sel2.button("Deseleziona pagina", width="stretch", disabled=not n_visibili_selezionati):
            salva_selezione_galleria(selezionati - ids_visibili, aggiorna_checkbox=True)
            st.rerun()
        if col_sel3.button("Svuota selezione", width="stretch", disabled=not selezionati):
            salva_selezione_galleria([], aggiorna_checkbox=True)
            st.rerun()
        if col_zip.button("Prepara ZIP", width="stretch", disabled=not selezionati):
            with st.spinner("Creazione dello ZIP degli originali..."):
                try:
                    inclusi, mancanti = prepara_zip_galleria(selezionati)
                    if inclusi:
                        st.success(f"ZIP pronto con {len(inclusi)} elementi.")
                    if mancanti:
                        st.warning(f"{len(mancanti)} originali non sono stati trovati e sono stati esclusi.")
                except Exception as errore:
                    st.error(f"Impossibile creare lo ZIP: {errore}")
        if col_del.button("Elimina selezionati", type="primary", width="stretch", disabled=not selezionati):
            richiedi_conferma_eliminazione(selezionati)

        percorso_zip = st.session_state.get("galleria_zip_path")
        if percorso_zip and os.path.exists(percorso_zip):
            mancanti_zip = st.session_state.get("galleria_zip_mancanti", [])
            if mancanti_zip:
                st.warning(f"ZIP creato senza {len(mancanti_zip)} originali non trovati.")
            with open(percorso_zip, "rb") as dati_zip:
                st.download_button(
                    "⬇️ Scarica originali selezionati (.zip)",
                    data=dati_zip,
                    file_name="deepsight-selezione.zip",
                    mime="application/zip",
                    key="gal_download_zip",
                )

        if not elementi_filtrati:
            st.info("Nessun elemento corrisponde alla ricerca e ai filtri selezionati.")
        else:
            # Container con key: il CSS lo trasforma in una griglia responsive e ritaglia le foto.
            with st.container(key="galleria_griglia"):
                for elemento in elementi_visibili:
                    with st.container():
                        chiave_checkbox = f"{PREFISSO_CHECKBOX_GALLERIA}{elemento['id']}"
                        st.session_state[chiave_checkbox] = elemento["id"] in selezionati
                        st.checkbox(
                            "Seleziona",
                            key=chiave_checkbox,
                            on_change=aggiorna_selezione_da_checkbox,
                            args=(elemento["id"],),
                        )
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-meta">
                                <div class="result-title" title="{elemento['filename']}">{elemento['filename']}</div>
                                <div style="opacity:0.7; font-size:0.8rem;">
                                    Tipo: {elemento['media_type'].upper()} | {elemento['width']}x{elemento['height']} <br>
                                    📅 {elemento['creation_date'].split('T')[0] if elemento['creation_date'] else 'N/D'} &nbsp;·&nbsp; 📍 {elemento.get('location_name') or 'N/D'}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if elemento["media_type"] == "image" and os.path.exists(elemento["file_path"]):
                            st.image(immagine_per_display(elemento["file_path"], lato_max=1000), width="stretch")
                        else:
                            percorso_anteprima = percorso_anteprima_elemento(elemento["file_path"])
                            if percorso_anteprima:
                                st.image(percorso_anteprima, width="stretch")

                        with st.expander("Azioni e Dettagli"):
                            st.write(f"**Percorso file:** `{elemento['file_path']}`")
                            st.write(f"**Dimensione:** {ottieni_stringa_dimensione_file(elemento['file_size'] or 0)}")

                            if elemento["media_type"] == "video" and os.path.exists(elemento["file_path"]):
                                mostra_player_video(elemento["file_path"])

                            if os.path.exists(elemento["file_path"]):
                                with open(elemento["file_path"], "rb") as dati_file:
                                    st.download_button(
                                        label="⬇️ Scarica File Originale",
                                        data=dati_file,
                                        file_name=elemento["filename"],
                                        mime="application/octet-stream",
                                        key=f"gal_dl_{elemento['id']}"
                                    )

                            if st.button("🗑️ Elimina dall'Archivio", key=f"gal_del_{elemento['id']}"):
                                richiedi_conferma_eliminazione([elemento["id"]])

            # Ultimo blocco della pagina: nessun controllo di paginazione compare in cima.
            mostra_paginatore_galleria(
                pagina_corrente,
                totale_pagine,
                totale_elementi,
                inizio_pagina,
                fine_pagina,
                scorrimento_continuo,
            )


# --- 3. SCHEDA CARICAMENTO & IMPORT ---
elif menu == "📤 Caricamento & Import":
    st.markdown("<h1 class='main-title'>Caricamento Nuovi Contenuti</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Carica singoli file o scansiona una cartella condivisa locale per indicizzarli</p>", unsafe_allow_html=True)
    
    # La modalità (upload file / scansione cartella) è scelta dall'opzione
    # premuta nel menu del pulsante flottante "+".
    modalita_caricamento = st.session_state.get("modalita_caricamento", "file")

    if modalita_caricamento == "file":
        file_caricati = st.file_uploader(
            "Seleziona Immagini o Video da importare", 
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv", "gif"],
            accept_multiple_files=True
        )
        
        if file_caricati:
            st.markdown(f"**Selezionati {len(file_caricati)} file.** Clicca sul pulsante in basso per avviare l'elaborazione locale.")
            if st.button("Elabora e Aggiungi all'Archivio"):
                area_log = st.empty()

                # FASE 1 (sincrona, veloce): registra subito TUTTO il lotto (copia in
                # archivio + record processed=0). Così il conteggio "file in coda"
                # della dashboard riflette l'intero caricamento.
                lotto = []  # (nome_file, id_media, percorso_archiviato, tipo_media)
                with st.spinner(f"Registrazione di {len(file_caricati)} file nell'archivio..."):
                    for file_oggetto in file_caricati:
                        # Salva il file temporaneamente su disco per passarlo alla pipeline
                        cartella_temporanea = tempfile.gettempdir()
                        percorso_temporaneo = os.path.join(cartella_temporanea, file_oggetto.name)
                        try:
                            with open(percorso_temporaneo, "wb") as f:
                                f.write(file_oggetto.read())
                            lotto.append((file_oggetto.name,) + processor.registra_file(percorso_temporaneo))
                        except Exception as errore:
                            area_log.error(f"Registrazione fallita per {file_oggetto.name}: {errore}")
                        finally:
                            # La copia in archivio è già stata fatta: il temporaneo non serve più
                            if os.path.exists(percorso_temporaneo):
                                os.remove(percorso_temporaneo)

                # FASE 2 (in background): il lavoratore della coda riprende da solo gli
                # elementi appena registrati (processed=0) e la pagina torna subito libera.
                if lotto:
                    processor.avvia_lavoratore()
                    st.success(
                        f"{len(lotto)} file registrati: elaborazione avviata in background. "
                        "Puoi continuare a navigare, l'avanzamento e i controlli pausa/riprendi sono nella barra laterale."
                    )
                else:
                    st.error("Nessun file registrato: controlla gli errori sopra.")
                
    else:
        st.markdown("""
        ### Scansione di una cartella di rete o locale
        Copia manualmente le immagini e i video in una cartella specifica. Inserisci il percorso qui sotto e l'applicazione scansionerà e indicizzerà i file non ancora elaborati.
        """)
        percorso_cartella_condivisa = st.text_input("Percorso Cartella Condivisa (es. C:\\ArchivioCondiviso)", "")
        
        if st.button("Avvia Scansione Cartella"):
            if not percorso_cartella_condivisa or not os.path.exists(percorso_cartella_condivisa):
                st.error("Il percorso inserito non è valido o non esiste.")
            else:
                with st.spinner("Scansione in corso..."):
                    conteggio_successi, conteggio_fallimenti = processor.scansiona_cartella_condivisa(percorso_cartella_condivisa)
                    st.success(f"Scansione completata! Nuovi file elaborati con successo: {conteggio_successi}. File falliti: {conteggio_fallimenti}.")


# --- SCHEDA PERSONE (Face Database) ---
elif menu == "👤 Persone":
    st.markdown("<h1 class='main-title'>Persone</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Tutte le persone riconosciute nell'archivio. Clicca una persona per vedere i suoi contenuti e assegnarle un nome.</p>", unsafe_allow_html=True)

    import persone as modulo_persone
    lista_persone = database.ottieni_persone()

    if st.button("🔄 Ricalcola raggruppamenti", key="btn_recluster",
                 help="Riesegue il clustering di tutti i volti (i nomi vengono preservati)"):
        with st.spinner("Re-clustering dei volti in corso..."):
            n = modulo_persone.ricalcola_tutti_cluster()
        st.toast(f"Raggruppamento completato: {n} persone.", icon="✅")
        st.rerun()

    if not lista_persone:
        st.info("Nessuna persona: carica contenuti e attendi lo stadio 'volti' della coda, "
                "oppure usa 🔄 Ricalcola raggruppamenti se l'archivio contiene già volti.")
    else:
        id_selezionata = st.session_state.get("persona_selezionata")

        if id_selezionata is None:
            # griglia delle persone
            with st.container(key="risultati_griglia"):
                for p in lista_persone:
                    with st.container():
                        nome = p["name"] or f"Persona {p['id']}"
                        st.markdown(f"""
                        <div class="result-card"><div class="result-meta">
                            <div class="result-title">{nome}</div>
                            <div style="opacity:0.7; font-size:0.8rem;">
                                {p['n_media']} contenuti · {p['n_volti']} volti</div>
                        </div></div>""", unsafe_allow_html=True)
                        if p["crop_path"] and os.path.exists(p["crop_path"]):
                            st.image(p["crop_path"], width="stretch")
                        if st.button("Apri", key=f"apri_persona_{p['id']}", width='stretch'):
                            st.session_state["persona_selezionata"] = p["id"]
                            st.rerun()
        else:
            persona = next((p for p in lista_persone if p["id"] == id_selezionata), None)
            if persona is None:
                st.session_state.pop("persona_selezionata", None)
                st.rerun()
            if st.button("← Tutte le persone", key="btn_indietro_persone"):
                st.session_state.pop("persona_selezionata", None)
                st.rerun()

            col_p1, col_p2 = st.columns([1, 2])
            with col_p1:
                if persona["crop_path"] and os.path.exists(persona["crop_path"]):
                    st.container(key="volti_query").image(persona["crop_path"], width=160)
            with col_p2:
                nuovo_nome = st.text_input("Nome", value=persona["name"] or "", key=f"nome_p_{persona['id']}")
                if st.button("💾 Salva nome", key=f"salva_nome_{persona['id']}"):
                    database.rinomina_persona(persona["id"], nuovo_nome.strip())
                    st.toast("Nome salvato.", icon="✅")
                    st.rerun()
                altre = [p for p in lista_persone if p["id"] != persona["id"]]
                if altre:
                    etichette = {f"{p['name'] or 'Persona ' + str(p['id'])} (#{p['id']})": p["id"] for p in altre}
                    scelta = st.selectbox("Unisci con...", ["—"] + list(etichette), key=f"merge_sel_{persona['id']}")
                    if scelta != "—" and st.button("🔗 Unisci (i volti passano a questa persona)", key=f"merge_btn_{persona['id']}"):
                        database.unisci_persone(etichette[scelta], persona["id"])
                        st.toast("Persone unite.", icon="✅")
                        st.rerun()

            nome_persona = persona["name"] or f"Persona {persona['id']}"
            st.markdown(f"### Contenuti con {nome_persona}")
            media_persona = database.ottieni_media_di_persona(persona["id"])
            with st.container(key="galleria_griglia"):
                for elemento in media_persona:
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card"><div class="result-meta">
                            <div class="result-title" title="{elemento['filename']}">{elemento['filename']}</div>
                            <div style="opacity:0.7; font-size:0.8rem;">
                                Tipo: {elemento['media_type'].upper()}<br>
                                📅 {elemento['creation_date'].split('T')[0] if elemento['creation_date'] else 'N/D'}</div>
                        </div></div>""", unsafe_allow_html=True)
                        if elemento["media_type"] == "image" and os.path.exists(elemento["file_path"]):
                            st.image(immagine_per_display(elemento["file_path"], lato_max=1000), width="stretch")
                        else:
                            anteprima = percorso_anteprima_elemento(elemento["file_path"])
                            if anteprima:
                                st.image(anteprima, width="stretch")
                        if os.path.exists(elemento["file_path"]):
                            with open(elemento["file_path"], "rb") as dati_file:
                                st.download_button("⬇️ Scarica", data=dati_file,
                                                   file_name=elemento["filename"],
                                                   mime="application/octet-stream",
                                                   key=f"persona_dl_{persona['id']}_{elemento['id']}")


# --- 4. SCHEDA RICERCA AVANZATA ---
elif menu == "🔍 Ricerca Avanzata":
    st.markdown("<h1 class='main-title'>Motore di Ricerca AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Cerca nel tuo archivio con semantica testuale, similarità visiva o volti</p>", unsafe_allow_html=True)

    statistiche_ricerca = database.ottieni_statistiche_db()
    if statistiche_ricerca["total_items"] == 0:
        st.warning(
            "L'archivio è vuoto. Carica almeno una foto, un video o una GIF "
            "prima di eseguire una ricerca."
        )
        if st.button("📤 Vai al caricamento", key="ricerca_vuota_carica", type="primary"):
            st.session_state.selezione_menu = "📤 Caricamento & Import"
            st.session_state.modalita_caricamento = "file"
            st.rerun()
        # Non inizializzare Qwen né interrogare Chroma quando non esistono contenuti.
        st.stop()

    if database.conteggio_media_cercabili() == 0:
        st.info(
            "I contenuti dell'archivio non sono ancora pronti per la ricerca. "
            "Attendi il completamento dell'elaborazione indicata nella barra laterale."
        )
        st.stop()

    # --- FILTRI DI RICERCA (applicati ai risultati di tutte le modalità) ---
    with st.expander("🎛️ Filtri di ricerca", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtro_tipo_media = st.selectbox("Tipo di file", ["Tutti", "Immagini", "Video"], key="sb_media_type")
        with col_f2:
            filtro_localita = st.text_input("Luogo di creazione", "", key="sb_location")

        abilita_filtro_date = st.checkbox("Filtra per data", key="sb_enable_date")
        filtro_data_inizio = None
        filtro_data_fine = None
        if abilita_filtro_date:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                filtro_data_inizio = st.date_input("Data Inizio", datetime.date.today() - datetime.timedelta(days=365), key="sb_date_start")
            with col_d2:
                filtro_data_fine = st.date_input("Data Fine", datetime.date.today(), key="sb_date_end")

    filtri = {
        "media_type": filtro_tipo_media,
        "location": filtro_localita,
        "date_start": filtro_data_inizio,
        "date_end": filtro_data_fine,
    }

    # Sotto-schede per le diverse modalità di ricerca
    sotto_scheda1, sotto_scheda2, sotto_scheda3, sotto_scheda4 = st.tabs([
        "🔍 Semantica (Testo)", 
        "🖼️ Similarità Immagine", 
        "👤 Riconoscimento Volto",
        "🗣️ Parlato (Whisper)"
    ])
    
    risultati = [] # Conterrà tuple: (elemento, punteggio, tipo_ricerca)
    
    with sotto_scheda1:
        st.markdown("#### Ricerca per concetto testuale")
        testo_query = st.text_input("Inserisci cosa stai cercando (es: 'una spiaggia al tramonto', 'strada con auto', 'cane che corre')", "")
        usa_negativo = st.toggle("Prompt negativo (escludi elementi)", key="tgl_negativo")
        testo_negativo = ""
        if usa_negativo:
            testo_negativo = st.text_input("Cosa NON deve comparire (es: 'persone', 'neve')", "", key="txt_negativo")

        if testo_query:
            with st.spinner("Calcolo embedding della query e ricerca vettoriale..."):
                try:
                    qwen = gestore.ottieni_qwen()
                    query_emb = qwen.ottieni_embedding_testo(testo_query)
                    emb_negativo = None
                    if usa_negativo and testo_negativo.strip():
                        emb_negativo = qwen.ottieni_embedding_testo(testo_negativo.strip())

                    # Nessuna soglia: si mostrano sempre i migliori per rilevanza
                    # (top-5 + "Mostra altri"), cosi' una query non resta mai a vuoto.
                    for f, sim in database.cerca_frame_simili(query_emb):
                        if not applica_filtri(f, filtri):
                            continue
                        punteggio = sim
                        if emb_negativo is not None:
                            # penalita': gli elementi affini al prompt negativo scendono
                            punteggio = sim - config.LAMBDA_PROMPT_NEGATIVO * float(
                                np.dot(emb_negativo, f["embedding"]))
                        risultati.append((f, punteggio, "clip"))

                    risultati = deduplica_risultati(risultati)
                except Exception as errore:
                    st.error(f"Errore nella ricerca semantica: {errore}")

    with sotto_scheda2:
        st.markdown("#### Trova elementi visivamente simili a un'immagine query")
        file_immagine_query = st.file_uploader("Carica un'immagine di esempio", type=["jpg", "jpeg", "png"], key="img_sim")
        
        if file_immagine_query:
            immagine_pil_query = ImageOps.exif_transpose(Image.open(file_immagine_query)).convert("RGB")
            # Container con key: il CSS ne toglie il pulsante fullscreen (immagine di lavoro,
            # non un elemento d'archivio), come per i ritagli dei volti.
            st.container(key="immagine_query").image(immagine_pil_query, caption="Immagine di Query", width=200)

            id_file_img = getattr(file_immagine_query, "file_id", None) or file_immagine_query.name
            # Al click l'embedding della query viene calcolato UNA volta e memorizzato
            # (con l'id del file). La ricerca vera e propria viene poi rieseguita a OGNI
            # rerun finché è caricata la stessa immagine: così i risultati restano
            # presenti anche dopo un rerun causato da altri pulsanti (es. "Elimina"),
            # altrimenti l'eliminazione non verrebbe mai eseguita.
            if st.button("Esegui Ricerca per Similarità"):
                with st.spinner("Elaborazione immagine di query..."):
                    qwen = gestore.ottieni_qwen()
                    st.session_state["emb_query_img"] = qwen.ottieni_embedding_immagine(immagine_pil_query)
                    st.session_state["id_file_img"] = id_file_img

            if st.session_state.get("emb_query_img") is not None and st.session_state.get("id_file_img") == id_file_img:
                with st.spinner("Ricerca vettoriale su ChromaDB..."):
                    try:
                        # Anche qui niente soglia (vedi ricerca testuale): ordina per
                        # similarita' e lascia al top-5 + "Mostra altri" il taglio.
                        for f, sim in database.cerca_frame_simili(st.session_state["emb_query_img"]):
                            if not applica_filtri(f, filtri):
                                continue
                            risultati.append((f, sim, "clip"))

                        risultati = deduplica_risultati(risultati)
                    except Exception as errore:
                        st.error(f"Errore nella ricerca per similarità d'immagine: {errore}")
                        
    with sotto_scheda3:
        st.markdown("#### Ricerca volti tramite embedding biometrici")
        file_volto_query = st.file_uploader("Carica una foto contenente il volto da cercare", type=["jpg", "jpeg", "png"], key="face_sim")
        
        if file_volto_query:
            immagine_pil_volto_query = ImageOps.exif_transpose(Image.open(file_volto_query)).convert("RGB")
            
            # Esegue il rilevamento dei volti sull'immagine caricata
            with st.spinner("Rilevamento volti nell'immagine caricata..."):
                face_rec = gestore.ottieni_volti()
                volti_rilevati = face_rec.rileva_e_codifica_volti(immagine_pil_volto_query)
                
            if not volti_rilevati:
                st.warning("Nessun volto rilevato nell'immagine fornita. Assicurati che il volto sia ben visibile e illuminato.")
            else:
                st.success(f"Rilevati {len(volti_rilevati)} volti!")
                
                # Consente la selezione se sono rilevati più volti
                # Il container con key permette al CSS di togliere il fullscreen dai ritagli.
                indice_volto_selezionato = 0
                contenitore_volti = st.container(key="volti_query")
                if len(volti_rilevati) > 1:
                    contenitore_volti.markdown("Seleziona quale volto desideri cercare nell'archivio:")
                    colonne = contenitore_volti.columns(len(volti_rilevati))
                    opzioni = []
                    for idx, fd in enumerate(volti_rilevati):
                        colonne[idx].image(fd["crop"], width=100)
                        opzioni.append(f"Volto {idx+1} (Confidenza: {fd['confidence']:.2%})")

                    opzione_selezionata = contenitore_volti.selectbox("Seleziona Volto", opzioni)
                    indice_volto_selezionato = opzioni.index(opzione_selezionata)
                else:
                    contenitore_volti.image(volti_rilevati[0]["crop"], caption="Volto rilevato", width=120)
                    
                id_file_volto = getattr(file_volto_query, "file_id", None) or file_volto_query.name
                # Come per la ricerca immagine: al click si memorizza l'embedding del volto
                # scelto; la ricerca viene poi rieseguita a ogni rerun finché è caricata la
                # stessa foto, così i risultati restano e il pulsante "Elimina" funziona.
                if st.button("Cerca Volto nell'Archivio"):
                    st.session_state["emb_query_volto"] = volti_rilevati[indice_volto_selezionato]["embedding"]
                    st.session_state["id_file_volto"] = id_file_volto

                if st.session_state.get("emb_query_volto") is not None and st.session_state.get("id_file_volto") == id_file_volto:
                    with st.spinner("Confronto dei profili facciali con ChromaDB..."):
                        for f, sim in database.cerca_volti_simili(st.session_state["emb_query_volto"]):
                            if not applica_filtri(f, filtri):
                                continue
                            if sim >= config.SOGLIA_SIMILARITA_VOLTI:
                                risultati.append((f, sim, "face"))

                        risultati = deduplica_risultati(risultati)
                        risultati = risultati[:30]
                        
    with sotto_scheda4:
        st.markdown("#### Cerca audio parlato nei video (Whisper)")
        parola_chiave = st.text_input("Inserisci parole chiave da cercare (es. parole pronunciate)", "")

        if parola_chiave:
            with st.spinner("Ricerca testuale nel database..."):
                try:
                    connessione = database.ottieni_connessione()
                    cursore = connessione.cursor()

                    # Ricerca nelle trascrizioni audio di Whisper (Video)
                    cursore.execute("""
                        SELECT id, file_path, filename, media_type, creation_date, location_name, transcription
                        FROM media_items
                        WHERE media_type = 'video' AND processed = 1 AND transcription LIKE ?
                    """, (f"%{parola_chiave}%",))
                    righe_video = cursore.fetchall()
                    connessione.close()

                    # Converte le corrispondenze video
                    for r in righe_video:
                        nome_file = r[2]
                        # Miniatura per nome-hash; se assente resta None (mai il file
                        # video stesso: st.image non saprebbe visualizzarlo).
                        percorso_anteprima = percorso_anteprima_elemento(r[1])

                        elemento = {
                            "frame_id": None,
                            "media_id": r[0],
                            "frame_index": 0,
                            "timestamp_seconds": 0.0,
                            "image_path": percorso_anteprima,
                            "objects": [],
                            "filename": nome_file,
                            "file_path": r[1],
                            "media_type": r[3],
                            "creation_date": r[4],
                            "location_name": r[5] or "",
                            "matched_type": "Parlato Video"
                        }
                        if applica_filtri(elemento, filtri):
                            # Punteggio fisso 1.0 per corrispondenza testuale esatta
                            risultati.append((elemento, 1.0, "text_whisper"))

                    risultati = deduplica_risultati(risultati)
                    risultati = risultati[:30]

                except Exception as errore:
                    st.error(f"Errore nella ricerca testuale: {errore}")

    # --- MOSTRA RISULTATI DELLA RICERCA ---
    if risultati:
        # Si mostrano solo i migliori N per rilevanza; "Mostra altri" allarga la finestra.
        # Il contatore riparte da 5 quando cambiano i criteri di ricerca (firma).
        firma_ricerca = (testo_query, testo_negativo,
                         st.session_state.get("id_file_img"),
                         st.session_state.get("id_file_volto"), parola_chiave)
        if st.session_state.get("firma_ricerca") != firma_ricerca:
            st.session_state["firma_ricerca"] = firma_ricerca
            st.session_state["n_risultati_mostrati"] = 5
        n_mostrati = min(st.session_state.get("n_risultati_mostrati", 5), len(risultati))
        st.markdown(f"### Risultati della Ricerca (i migliori {n_mostrati} di {len(risultati)})")

        # Container con key: il CSS lo trasforma in una griglia responsive e ritaglia le foto
        # in quadrati uniformi. Un container per elemento = una cella della griglia.
        griglia_risultati = st.container(key="risultati_griglia")

        for idx, corrispondenza in enumerate(risultati[:n_mostrati]):
            elemento, punteggio, modalita = corrispondenza

            with griglia_risultati.container():
                # Decide l'immagine da mostrare. La card mostra sempre il media ORIGINALE, anche
                # per i risultati volto: chi cerca un volto vuole ritrovare la foto/il video in cui
                # compare, non il ritaglio (che resta in "Azioni e Dettagli").
                # I risultati volto non hanno "image_path": .get() evita il KeyError e il fallback
                # qui sotto risolve sulla miniatura dell'originale.
                immagine_da_mostrare = elemento.get("image_path")

                # Fallback se la miniatura specifica non esiste (lookup per nome-hash)
                if not immagine_da_mostrare or not os.path.exists(immagine_da_mostrare):
                    anteprima_alternativa = percorso_anteprima_elemento(elemento["file_path"])
                    if anteprima_alternativa:
                        immagine_da_mostrare = anteprima_alternativa
                    else:
                        immagine_da_mostrare = elemento["file_path"] if elemento["media_type"] == 'image' else None
                
                # Costruisce la stringa del punteggio o tipo. Discrimina sulla MODALITÀ, non sul
                # valore: le ricerche testuali (parlato) assegnano un punteggio fisso di 1.0,
                # ma anche un volto identico a uno in archivio ha coseno 1.0 e finirebbe
                # etichettato come corrispondenza testuale.
                stringa_punteggio = "Corrispondenza Testo" if modalita.startswith("text_") else f"{max(0.0, punteggio) * 100:.1f}% Rilevanza"
                
                # Stringa dei dettagli (es. timestamp per i video)
                stringa_dettagli = ""
                if elemento["media_type"] == "video":
                    timestamp_formattato = str(datetime.timedelta(seconds=int(elemento["timestamp_seconds"])))
                    stringa_dettagli = f"⏰ Frame Video a: <code>{timestamp_formattato}</code>"
                elif modalita == "face":
                    stringa_dettagli = "👤 Volto rilevato nell'immagine"
                else:
                    stringa_dettagli = "🖼️ Immagine"
                
                # Rendering HTML della scheda dei risultati
                st.markdown(f"""
                <div class="result-card">
                    <div style="background-color: var(--superficie-2); padding: var(--spazio-3); border-bottom: 1px solid var(--bordo);">
                        <span class="result-score">{stringa_punteggio}</span>
                        <div class="result-title" title="{elemento['filename']}">{elemento['filename']}</div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">
                            {stringa_dettagli}<br>
                            📅 Data: {elemento['creation_date'].split('T')[0] if elemento['creation_date'] else 'N/D'} | 📍 Luogo: {elemento['location_name'] or 'N/D'}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if immagine_da_mostrare and os.path.exists(immagine_da_mostrare):
                    st.image(immagine_per_display(immagine_da_mostrare, lato_max=1000), width="stretch")
                
                # Dettagli aggiuntivi espandibili
                with st.expander("Azioni e Dettagli"):
                    st.write(f"**Percorso file:** `{elemento['file_path']}`")

                    # Ritaglio del volto che ha prodotto la corrispondenza: nelle foto di gruppo
                    # dice QUALE dei volti presenti ha fatto match.
                    if modalita == "face" and elemento.get("crop_path") and os.path.exists(elemento["crop_path"]):
                        contenitore_crop = st.container(key=f"crop_volto_{elemento['face_id']}_{idx}")
                        contenitore_crop.image(elemento["crop_path"], caption="Volto corrispondente", width=120)

                    if elemento.get("objects"):
                        st.markdown("**Tag rilevati:**")
                        tag_html = "".join([f'<span class="tag-pill">{tag}</span>' for tag in elemento["objects"]])
                        st.markdown(tag_html, unsafe_allow_html=True)
                        
                    # Player video dedicato se elemento è video (e il file esiste ancora:
                    # un video rimosso a mano dal disco manderebbe st.video in errore)
                    if elemento["media_type"] == "video" and os.path.exists(elemento["file_path"]):
                        st.write("---")
                        timestamp_formattato = str(datetime.timedelta(seconds=int(elemento["timestamp_seconds"])))
                        st.markdown(f"**Riproduci Video a {timestamp_formattato}:**")
                        mostra_player_video(elemento["file_path"], start_time=int(elemento["timestamp_seconds"]))
                        
                    # Bottone per scaricare il file originale
                    if os.path.exists(elemento["file_path"]):
                        with open(elemento["file_path"], "rb") as dati_file:
                            st.download_button(
                                label="⬇️ Scarica File Originale",
                                data=dati_file,
                                file_name=elemento["filename"],
                                mime="application/octet-stream",
                                key=f"dl_{elemento['media_id']}_{idx}"
                            )
                            
                    # Bottone per eliminare l'elemento dall'archivio
                    if st.button("🗑️ Elimina dall'Archivio", key=f"del_{elemento['media_id']}_{idx}"):
                        richiedi_conferma_eliminazione([elemento["media_id"]])

        if len(risultati) > n_mostrati:
            if st.button(f"➕ Mostra altri ({len(risultati) - n_mostrati} rimanenti)",
                         key="btn_mostra_altri", width='stretch'):
                st.session_state["n_risultati_mostrati"] = n_mostrati + 10
                st.rerun()
    else:
        if any([testo_query, file_immagine_query, file_volto_query, parola_chiave]):
            st.warning("Nessun risultato trovato corrispondente ai criteri di ricerca ed ai filtri inseriti.")
        else:
            st.info("Imposta un criterio di ricerca in alto per iniziare.")

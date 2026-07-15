#!/usr/bin/env bash
# Installatore DeepSight per Linux/macOS: crea il venv, rileva la GPU NVIDIA per
# scegliere la build corretta di PyTorch, installa le dipendenze. Equivalente di
# scripts/windows/install.ps1. A differenza dello script Windows NON installa
# Python automaticamente: le distribuzioni Linux differiscono troppo (apt/dnf/
# pacman/zypper) per farlo in modo affidabile: si limita a indicare il comando.
set -euo pipefail

# Si posiziona nella cartella radice del progetto (questo script vive in
# ./scripts/linux/, quindi risale di due livelli), cosi' venv/requirements.txt/src
# vengono trovati indipendentemente da dove viene lanciato lo script.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

VERDE='\033[0;32m'; GIALLO='\033[1;33m'; ROSSO='\033[0;31m'; CIANO='\033[0;36m'; RESET='\033[0m'

echo -e "${CIANO}==========================================================${RESET}"
echo -e "${CIANO}   DeepSight - SCRIPT DI INSTALLAZIONE LOCALE (Linux)${RESET}"
echo -e "${CIANO}==========================================================${RESET}"

# 1. Verifica presenza di Python >= 3.11 (stessa soglia minima di install.ps1:
#    numpy>=2.0, chromadb e i wheel di PyTorch non funzionano su Python obsoleti).
PYTHON_VERSIONE_MINIMA="3.11"

versione_ok() {
    # Confronta "$1" (es. 3.13) >= 3.11 senza dipendere da `sort -V` o `bc`.
    local maggiore="${1%%.*}" minore="${1#*.}"
    local maggiore_min="${PYTHON_VERSIONE_MINIMA%%.*}" minore_min="${PYTHON_VERSIONE_MINIMA#*.}"
    [ "$maggiore" -gt "$maggiore_min" ] || { [ "$maggiore" -eq "$maggiore_min" ] && [ "$minore" -ge "$minore_min" ]; }
}

python_exe=""
for candidato in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidato" >/dev/null 2>&1; then
        v=$("$candidato" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || true)
        if [ -n "$v" ] && versione_ok "$v"; then
            python_exe="$candidato"
            break
        fi
    fi
done

if [ -z "$python_exe" ]; then
    echo -e "${ROSSO}ERRORE: nessun Python $PYTHON_VERSIONE_MINIMA+ trovato.${RESET}"
    echo -e "${GIALLO}Installalo con il gestore pacchetti della tua distribuzione, ad esempio:${RESET}"
    echo "  Ubuntu/Debian:  sudo apt install python3 python3-venv python3-pip"
    echo "  Fedora:         sudo dnf install python3 python3-pip"
    echo "  Arch:           sudo pacman -S python python-pip"
    echo "Poi rilancia questo script."
    exit 1
fi

echo -e "${VERDE}Rilevato: $("$python_exe" --version)${RESET}"

# 1-bis. Verifica che il modulo venv sia utilizzabile. Su Debian/Ubuntu Python e'
# spezzato in piu' pacchetti: 'python3' c'e' e supera il controllo di versione qui
# sopra, ma 'ensurepip' vive in python3-venv e senza quello la creazione dell'
# ambiente virtuale fallisce. Meglio dirlo subito, con il nome del pacchetto giusto.
if ! "$python_exe" -c "import venv, ensurepip" >/dev/null 2>&1; then
    echo -e "${ROSSO}ERRORE: il modulo 'venv' di Python non e' utilizzabile (manca 'ensurepip').${RESET}"
    echo -e "${GIALLO}Su Debian/Ubuntu va installato a parte:${RESET}"
    # Su Ubuntu il pacchetto e' spesso versionato (es. python3.12-venv), quindi si
    # suggerisce prima quello che corrisponde all'interprete effettivamente trovato.
    echo "  sudo apt install python${v}-venv    # oppure: sudo apt install python3-venv"
    echo -e "${GIALLO}Su Fedora e Arch 'venv' e' gia' incluso nel pacchetto Python.${RESET}"
    echo "Poi rilancia questo script."
    exit 1
fi

# 2. Creazione dell'ambiente virtuale venv
echo -e "\n${CIANO}[1/6] Creazione dell'ambiente virtuale (venv) in corso...${RESET}"
"$python_exe" -m venv venv
echo -e "${VERDE}Ambiente virtuale creato con successo!${RESET}"

# 3. Aggiornamento pip interno
echo -e "\n${CIANO}[2/6] Aggiornamento di pip nell'ambiente virtuale...${RESET}"
./venv/bin/python -m pip install --upgrade pip

# 4. Rilevamento GPU NVIDIA per scegliere la build corretta di PyTorch
echo -e "\n${CIANO}[3/6] Rilevamento hardware (GPU NVIDIA)...${RESET}"
if command -v nvidia-smi >/dev/null 2>&1; then
    indice_torch="https://download.pytorch.org/whl/cu124"
    echo -e "${VERDE}GPU NVIDIA rilevata: verra' installato PyTorch con supporto CUDA (cu124).${RESET}"
else
    indice_torch="https://download.pytorch.org/whl/cpu"
    echo -e "${GIALLO}Nessuna GPU NVIDIA rilevata: verra' installato PyTorch in versione CPU (download piu' leggero).${RESET}"
fi

# 5. Installazione di PyTorch dalla build corretta (CUDA o CPU)
echo -e "\n${CIANO}[4/6] Installazione di PyTorch (torch, torchvision)...${RESET}"
echo -e "${GIALLO}Questa operazione potrebbe richiedere diversi minuti a seconda della connessione internet...${RESET}"
./venv/bin/python -m pip install torch torchvision --index-url "$indice_torch"

# 6. Installazione degli altri pacchetti base da requirements.txt
#    (torch/torchvision gia' installati dalla index CUDA/CPU corretta: pip li
#    vedra' gia' soddisfatti. Per aggiungere una dipendenza basta requirements.txt,
#    senza toccare questo script.)
echo -e "\n${CIANO}[5/6] Installazione dei pacchetti base da requirements.txt (Whisper, Streamlit, OpenCV, ChromaDB)...${RESET}"
./venv/bin/python -m pip install -r requirements.txt

# 7. Installazione modelli local-only no-deps
echo -e "\n${CIANO}[6/6] Installazione di facenet-pytorch (--no-deps, per evitare NumPy < 2.0)...${RESET}"
./venv/bin/python -m pip install facenet-pytorch --no-deps

echo -e "\n${VERDE}==========================================================${RESET}"
echo -e "${VERDE} INSTALLAZIONE COMPLETATA CON SUCCESSO!${RESET}"
echo -e "${VERDE}==========================================================${RESET}"
echo -e "${GIALLO}Per avviare l'applicazione: ./scripts/linux/run.sh${RESET}"
echo -e "${VERDE}==========================================================${RESET}"

# 8. Download automatico dei modelli Qwen (~2.1 GB da Hugging Face) e di
#    llama-server (release ufficiale llama.cpp, build CPU). Idempotente: i file
#    gia' presenti non vengono riscaricati. Non bloccante (set +e locale): se il
#    download fallisce l'app parte comunque e la coda segnera' gli embedding
#    come falliti, recuperabili con "Riprova falliti" dopo aver rieseguito lo script.
echo -e "\n${CIANO}[Extra] Download modelli Qwen + llama-server (se mancanti)...${RESET}"
dir_qwen="models/qwen"
mkdir -p "$dir_qwen"
set +e

# Scarica su file .part e rinomina a fine download: un download interrotto non
# lascia mai un file parziale col nome definitivo (che verrebbe saltato al retry).
scarica() { # $1=url $2=destinazione
    if [ -f "$2" ]; then echo -e "${VERDE}  gia' presente: $(basename "$2")${RESET}"; return 0; fi
    echo -e "${CIANO}  download: $(basename "$2")${RESET}"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -C - -o "$2.part" "$1"
    else
        wget -c -O "$2.part" "$1"
    fi
    if [ $? -ne 0 ]; then echo -e "${ROSSO}  FALLITO: $(basename "$2")${RESET}"; return 1; fi
    mv -f "$2.part" "$2"
}

url_hf="https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF/resolve/main"
ok=0
scarica "$url_hf/Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf" "$dir_qwen/Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf" || ok=1
scarica "$url_hf/mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf" "$dir_qwen/mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf" || ok=1

# llama-server: release pinnata di llama.cpp (build CPU: l'app lo lancia con -ngl 0).
# ponytail: versione fissa b10016, da alzare a mano se una futura quantizzazione la richiede.
if [ ! -f "$dir_qwen/llama-server" ]; then
    tag_llama="b10016"
    tar_llama="/tmp/llama-$tag_llama-bin-ubuntu-x64.tar.gz"
    if scarica "https://github.com/ggml-org/llama.cpp/releases/download/$tag_llama/llama-$tag_llama-bin-ubuntu-x64.tar.gz" "$tar_llama"; then
        dir_estrazione="/tmp/llama-estratto"
        rm -rf "$dir_estrazione"; mkdir -p "$dir_estrazione"
        tar -xzf "$tar_llama" -C "$dir_estrazione"
        # Il layout interno dell'archivio e' cambiato tra le release: si cerca il
        # binario ovunque e si copiano eseguibile + librerie dalla sua cartella.
        exe=$(find "$dir_estrazione" -name llama-server -type f | head -n 1)
        if [ -n "$exe" ]; then
            cp "$(dirname "$exe")"/* "$dir_qwen/" 2>/dev/null
            chmod +x "$dir_qwen/llama-server"
            echo -e "${VERDE}  llama-server installato in $dir_qwen/${RESET}"
        else
            echo -e "${ROSSO}  FALLITO: llama-server non trovato nell'archivio.${RESET}"; ok=1
        fi
        rm -rf "$dir_estrazione" "$tar_llama"
    else ok=1; fi
else
    echo -e "${VERDE}  gia' presente: llama-server${RESET}"
fi
set -e

if [ "$ok" -ne 0 ]; then
    echo -e "\n${GIALLO}ATTENZIONE: download dei modelli incompleto. Riesegui questo script con${RESET}"
    echo -e "${GIALLO}una connessione attiva, oppure copia i file a mano in $dir_qwen/ (vedi README).${RESET}"
fi

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
echo -e "\n${CIANO}[5/6] Installazione dei pacchetti base da requirements.txt (CLIP, Whisper, Streamlit, OpenCV, ChromaDB)...${RESET}"
./venv/bin/python -m pip install -r requirements.txt

# 7. Installazione modelli local-only no-deps
echo -e "\n${CIANO}[6/6] Installazione di facenet-pytorch e easyocr (--no-deps, per evitare NumPy < 2.0)...${RESET}"
./venv/bin/python -m pip install facenet-pytorch easyocr --no-deps

echo -e "\n${VERDE}==========================================================${RESET}"
echo -e "${VERDE} INSTALLAZIONE COMPLETATA CON SUCCESSO!${RESET}"
echo -e "${VERDE}==========================================================${RESET}"
echo -e "${GIALLO}Per avviare l'applicazione: ./scripts/linux/run.sh${RESET}"
echo -e "${VERDE}==========================================================${RESET}"

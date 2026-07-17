#!/usr/bin/env bash
# Installazione riproducibile DeepSight per Linux x86-64.
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

VERDE='\033[0;32m'; GIALLO='\033[1;33m'; ROSSO='\033[0;31m'; CIANO='\033[0;36m'; RESET='\033[0m'
acceleratore="cpu"
prefetch_modelli=0

uso() {
    echo "Uso: $0 [--cpu|--cuda] [--prefetch-models]"
    echo "  --cpu             profilo più compatibile (default)"
    echo "  --cuda            PyTorch CUDA; Qwen resta CPU su Linux"
    echo "  --prefetch-models scarica anche FaceNet e Whisper durante l'installazione"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpu) acceleratore="cpu" ;;
        --cuda) acceleratore="cuda" ;;
        --prefetch-models) prefetch_modelli=1 ;;
        -h|--help) uso; exit 0 ;;
        *) echo "Argomento sconosciuto: $1"; uso; exit 2 ;;
    esac
    shift
done

printf "%b\n" "${CIANO}==========================================================${RESET}"
printf "%b\n" "${CIANO}  DeepSight - installazione Linux (${acceleratore})${RESET}"
printf "%b\n" "${CIANO}==========================================================${RESET}"

# Preflight: supporto ufficiale Ubuntu 22.04/24.04 x86-64. Altre distribuzioni
# possono proseguire perché lo stack è standard, ma ricevono un avviso esplicito.
if [ "$(uname -s)" != "Linux" ]; then
    printf "%b\n" "${ROSSO}ERRORE: questo script supporta Linux; macOS richiede un installer dedicato.${RESET}"
    exit 1
fi
architettura="$(uname -m)"
if [ "$architettura" != "x86_64" ] && [ "$architettura" != "amd64" ]; then
    printf "%b\n" "${ROSSO}ERRORE: architettura $architettura non supportata; serve x86-64.${RESET}"
    exit 1
fi

id_distro="sconosciuta"
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    id_distro="${ID:-sconosciuta}"
fi
case "$id_distro" in
    ubuntu|debian) ;;
    *) printf "%b\n" "${GIALLO}Avviso: $id_distro non è ancora validata ufficialmente; proseguo in modalità compatibile.${RESET}" ;;
esac

mancanti=()
for comando in tar sha256sum; do
    command -v "$comando" >/dev/null 2>&1 || mancanti+=("$comando")
done
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    mancanti+=("curl-o-wget")
fi
if [ "${#mancanti[@]}" -gt 0 ]; then
    printf "%b\n" "${ROSSO}ERRORE: utility mancanti: ${mancanti[*]}${RESET}"
    echo "Ubuntu/Debian: sudo apt install curl tar coreutils"
    exit 1
fi

spazio_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
if [ -n "$spazio_kb" ] && [ "$spazio_kb" -lt 8388608 ]; then
    printf "%b\n" "${GIALLO}Avviso: meno di 8 GB liberi; ambiente e modelli potrebbero non entrare.${RESET}"
fi

PYTHON_VERSIONE_MINIMA="3.11"
versione_ok() {
    local maggiore="${1%%.*}" minore="${1#*.}"
    [ "$maggiore" -gt 3 ] || { [ "$maggiore" -eq 3 ] && [ "$minore" -ge 11 ]; }
}

python_exe=""
for candidato in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidato" >/dev/null 2>&1; then
        versione=$("$candidato" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || true)
        if [ -n "$versione" ] && versione_ok "$versione"; then
            python_exe="$candidato"
            break
        fi
    fi
done
if [ -z "$python_exe" ]; then
    printf "%b\n" "${ROSSO}ERRORE: serve Python ${PYTHON_VERSIONE_MINIMA}+ (raccomandati 3.11/3.12).${RESET}"
    echo "Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
if ! "$python_exe" -c "import venv, ensurepip" >/dev/null 2>&1; then
    versione=$("$python_exe" -c "import sys; print('%d.%d' % sys.version_info[:2])")
    printf "%b\n" "${ROSSO}ERRORE: modulo venv/ensurepip non disponibile.${RESET}"
    echo "Ubuntu/Debian: sudo apt install python${versione}-venv"
    exit 1
fi
printf "%b\n" "${VERDE}Preflight OK: $id_distro $architettura, $("$python_exe" --version)${RESET}"

if [ -d venv ] && [ ! -x venv/bin/python ]; then
    printf "%b\n" "${ROSSO}ERRORE: venv esiste ma non è Linux (forse proviene da Windows).${RESET}"
    echo "Spostalo o eliminalo manualmente, poi rilancia l'installatore."
    exit 1
fi

printf "%b\n" "${CIANO}[1/6] Ambiente virtuale${RESET}"
"$python_exe" -m venv venv
PYTHON_VENV="./venv/bin/python"
"$PYTHON_VENV" -m pip install -q --upgrade pip setuptools wheel

printf "%b\n" "${CIANO}[2/6] PyTorch e torchvision${RESET}"
if [ "$acceleratore" = "cuda" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf "%b\n" "${ROSSO}ERRORE: --cuda richiesto ma nvidia-smi non è disponibile.${RESET}"
        exit 1
    fi
    indice_torch="${DEEPSIGHT_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
else
    indice_torch="https://download.pytorch.org/whl/cpu"
fi
"$PYTHON_VENV" -m pip install -q "torch>=2.0,<3.0" "torchvision>=0.15,<1.0" --index-url "$indice_torch"
printf "%b\n" "${VERDE}  OK torch + torchvision (${acceleratore})${RESET}"

installa_pacchetto() {
    local spec="$1"; shift
    local nome="${spec%%[<>=!~; ]*}"
    if "$PYTHON_VENV" -m pip install -q "$spec" "$@"; then
        printf "%b\n" "${VERDE}  OK $nome${RESET}"
    else
        printf "%b\n" "${ROSSO}  ERRORE installazione $nome${RESET}"
        exit 1
    fi
}

printf "%b\n" "${CIANO}[3/6] Dipendenze applicative${RESET}"
while IFS= read -r riga || [ -n "$riga" ]; do
    riga="${riga%$'\r'}"
    spec="${riga%%#*}"
    read -r spec <<< "$spec" || true
    [ -z "$spec" ] && continue
    nome="${spec%%[<>=!~; ]*}"
    [ "$nome" = torch ] && continue
    [ "$nome" = torchvision ] && continue
    installa_pacchetto "$spec"
done < requirements.txt
installa_pacchetto "facenet-pytorch==2.6.0" --no-deps

printf "%b\n" "${CIANO}[4/6] Modelli Qwen e llama-server${RESET}"
dir_qwen="models/qwen"
mkdir -p "$dir_qwen"

scarica() { # $1=url, $2=destinazione
    local url="$1" destinazione="$2"
    if [ -f "$destinazione" ]; then
        printf "%b\n" "${VERDE}  presente: $(basename "$destinazione")${RESET}"
        return 0
    fi
    printf "%b\n" "${CIANO}  download: $(basename "$destinazione")${RESET}"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -C - -o "$destinazione.part" "$url"
    else
        wget -c -O "$destinazione.part" "$url"
    fi
    mv -f "$destinazione.part" "$destinazione"
}

url_hf="https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF/resolve/main"
sha_atteso() {
    case "$1" in
        "Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf") echo "3f2f9023f15d5f3f084034eb5f14cc04a8e8d89b1f262354db9cf63c50308206" ;;
        "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf") echo "3f89a7768ffa6606935319f71bf56bb71871249ba549bf1080a0caea7a088613" ;;
    esac
}
for nome_gguf in "Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf" "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf"; do
    scarica "$url_hf/$nome_gguf" "$dir_qwen/$nome_gguf"
    sha_file=$(sha256sum "$dir_qwen/$nome_gguf" | cut -d' ' -f1)
    if [ "$sha_file" != "$(sha_atteso "$nome_gguf")" ]; then
        rm -f -- "$dir_qwen/$nome_gguf"
        printf "%b\n" "${ROSSO}ERRORE: hash non valido per $nome_gguf; file eliminato.${RESET}"
        exit 1
    fi
done

if [ ! -f "$dir_qwen/llama-server" ]; then
    tag_llama="b10016"
    temporanea=$(mktemp -d)
    pulisci_temporanea() { rm -rf -- "$temporanea"; }
    trap pulisci_temporanea EXIT
    archivio_llama="$temporanea/llama.tar.gz"
    scarica "https://github.com/ggml-org/llama.cpp/releases/download/$tag_llama/llama-$tag_llama-bin-ubuntu-x64.tar.gz" "$archivio_llama"
    mkdir -p "$temporanea/estratto"
    tar -xzf "$archivio_llama" -C "$temporanea/estratto"
    exe=$(find "$temporanea/estratto" -name llama-server -type f | head -n 1)
    if [ -z "$exe" ]; then
        printf "%b\n" "${ROSSO}ERRORE: llama-server non trovato nell'archivio ufficiale.${RESET}"
        exit 1
    fi
    cp -a "$(dirname "$exe")"/. "$dir_qwen"/
fi
chmod +x "$dir_qwen/llama-server"
if ! "$dir_qwen/llama-server" --version >/dev/null 2>&1; then
    printf "%b\n" "${ROSSO}ERRORE: llama-server non è avviabile (librerie Linux mancanti).${RESET}"
    exit 1
fi

printf "%b\n" "${CIANO}[5/6] Modelli opzionali${RESET}"
if [ "$prefetch_modelli" -eq 1 ]; then
    "$PYTHON_VENV" scripts/prefetch_models.py
else
    echo "  FaceNet e Whisper saranno scaricati al primo uso. Usa --prefetch-models per prepararli ora."
fi

printf "%b\n" "${CIANO}[6/6] Diagnostica finale${RESET}"
"$PYTHON_VENV" scripts/check_install.py
if [ "$acceleratore" = "cuda" ]; then
    "$PYTHON_VENV" -c "import torch; print('CUDA PyTorch disponibile:', torch.cuda.is_available())"
fi

printf "%b\n" "${VERDE}Installazione completata. Avvia con ./scripts/linux/run.sh${RESET}"

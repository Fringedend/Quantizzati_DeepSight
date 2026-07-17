#!/usr/bin/env bash
# Avvia DeepSight su Linux usando esclusivamente l'ambiente creato dall'installer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

VERDE='\033[0;32m'; GIALLO='\033[1;33m'; ROSSO='\033[0;31m'; CIANO='\033[0;36m'; RESET='\033[0m'

echo -e "${CIANO}==========================================================${RESET}"
echo -e "${CIANO}         DeepSight - AVVIO APPLICAZIONE${RESET}"
echo -e "${CIANO}==========================================================${RESET}"
echo ""

if [ -x "./venv/bin/python" ]; then
    echo -e "${VERDE}Ambiente virtuale 'venv' rilevato. Avvio applicazione...${RESET}"
    echo -e "${GIALLO}Il browser si aprira' automaticamente. Per chiudere, premi Ctrl+C.${RESET}"
    echo ""
    exec ./venv/bin/python -m streamlit run src/app.py
else
    echo -e "${ROSSO}ERRORE: ambiente virtuale Linux 'venv' non trovato.${RESET}"
    echo -e "${ROSSO}Esegui prima ./scripts/linux/install.sh per installare l'applicazione.${RESET}"
    exit 1
fi

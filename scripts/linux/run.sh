#!/usr/bin/env bash
# Avvia DeepSight su Linux/macOS. Equivalente di scripts/windows/run.ps1:
# usa l'interprete del venv se presente, altrimenti ripiega su streamlit globale.
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
    ./venv/bin/python -m streamlit run src/app.py
elif command -v streamlit >/dev/null 2>&1; then
    echo -e "${GIALLO}ATTENZIONE: ambiente virtuale 'venv' non trovato.${RESET}"
    echo -e "${GIALLO}Tentativo di avvio tramite installazione globale di streamlit...${RESET}"
    echo ""
    streamlit run src/app.py
else
    echo -e "${ROSSO}ERRORE: ambiente virtuale 'venv' non trovato e streamlit non e' nel PATH.${RESET}"
    echo -e "${ROSSO}Esegui prima ./scripts/linux/install.sh per installare l'applicazione.${RESET}"
    exit 1
fi

echo ""
echo -e "${GIALLO}==========================================================${RESET}"
echo -e "${GIALLO}  L'applicazione si e' chiusa.${RESET}"
echo -e "${GIALLO}==========================================================${RESET}"

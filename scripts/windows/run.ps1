# Imposta la directory di lavoro sulla cartella radice del progetto
# (questo script vive in .\scripts\windows\, quindi si risale di due livelli).
Set-Location -Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         DeepSight - AVVIO APPLICAZIONE" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# Verifica presenza cartella venv
if (Test-Path ".\venv\Scripts\python.exe") {
    Write-Host "Ambiente virtuale 'venv' rilevato. Avvio applicazione..." -ForegroundColor Green
    Write-Host "Il browser si aprira' automaticamente. Per chiudere, premi Ctrl+C." -ForegroundColor Yellow
    Write-Host ""

    try {
        # Esegue streamlit usando l'interprete python del venv
        & .\venv\Scripts\python.exe -m streamlit run src\app.py
    }
    catch {
        Write-Host ""
        Write-Host "ERRORE durante l'esecuzione di Streamlit:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
} else {
    Write-Host "ATTENZIONE: Ambiente virtuale 'venv' non trovato." -ForegroundColor Yellow
    Write-Host "Tentativo di avvio tramite installazione globale di python..." -ForegroundColor Yellow
    Write-Host ""

    # Verifica presenza streamlit nel path
    if (Get-Command "streamlit" -ErrorAction SilentlyContinue) {
        try {
            streamlit run src\app.py
        }
        catch {
            Write-Host ""
            Write-Host "ERRORE durante l'esecuzione di Streamlit:" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
    } else {
        Write-Host "ERRORE: Streamlit non e' installato o non e' nel PATH." -ForegroundColor Red
        Write-Host "Esegui prima scripts\windows\install.bat per installare l'applicazione." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Yellow
Write-Host "  L'applicazione si e' chiusa. Premi un tasto per uscire." -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Yellow
Pause

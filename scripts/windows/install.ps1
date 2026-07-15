# Imposta la directory di lavoro sulla cartella radice del progetto
# (questo script vive in .\scripts\windows\, quindi si risale di due livelli).
# Cosi' venv, requirements.txt e i sorgenti vengono trovati indipendentemente da
# dove viene lanciato lo script.
Set-Location -Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   DeepSight - SCRIPT DI INSTALLAZIONE LOCALE" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verifica presenza Python (con installazione automatica se assente o troppo vecchio)

# Versione minima richiesta: le dipendenze (numpy>=2.0, scikit-learn, chromadb, i wheel
# di PyTorch) non funzionano su Python obsoleti. Sotto questa soglia lo script ignora il
# Python presente e installa comunque la 3.13.
$PYTHON_VERSIONE_MINIMA = [version]"3.11"

# Ricarica la variabile PATH dal registro di sistema (utile subito dopo
# un'installazione, quando la sessione corrente non vede ancora i nuovi percorsi).
function Update-PathFromRegistry {
    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = ($machinePath, $userPath | Where-Object { $_ }) -join ';'
}

# Verifica che l'interprete indicato sia Python >= $PYTHON_VERSIONE_MINIMA.
function Test-PythonMinVersion {
    param([string]$exe)
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    try {
        $out = (& $exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and $out -match '^\d+\.\d+$') {
            return ([version]$out -ge $PYTHON_VERSIONE_MINIMA)
        }
    } catch {}
    return $false
}

# Individua un interprete Python 3 utilizzabile e ne restituisce il percorso, oppure $null.
function Resolve-Python {
    # Accetta solo interpreti >= $PYTHON_VERSIONE_MINIMA: un Python 3 troppo vecchio
    # viene ignorato, così più avanti scatta l'installazione automatica della 3.13.
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        # Evita l'alias fittizio dello Store, che aprirebbe il Microsoft Store.
        if (Test-PythonMinVersion $cmd.Source) { return $cmd.Source }
    }
    # Posizioni tipiche di installazione (per-utente e per-macchina).
    $candidati = @(
        "$env:LocalAppData\Programs\Python\Python313\python.exe",
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python313\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )
    foreach ($c in $candidati) { if (Test-PythonMinVersion $c) { return $c } }
    # Ultima risorsa: il launcher "py".
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $exe = (& $py.Source -3 -c "import sys; print(sys.executable)" 2>&1 | Out-String).Trim()
            if ($LASTEXITCODE -eq 0 -and (Test-PythonMinVersion $exe)) { return $exe }
        } catch {}
    }
    return $null
}

$pythonExe = Resolve-Python

if (-not $pythonExe) {
    Write-Host "Nessun Python $($PYTHON_VERSIONE_MINIMA.Major).$($PYTHON_VERSIONE_MINIMA.Minor)+ utilizzabile rilevato (assente o troppo vecchio): avvio l'installazione automatica di Python 3.13..." -ForegroundColor Yellow

    # Metodo 1: winget (disponibile su Windows 10/11 aggiornati).
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installazione di Python 3.13 tramite winget in corso..." -ForegroundColor Cyan
        winget install --id Python.Python.3.13 --source winget --scope user --silent --accept-package-agreements --accept-source-agreements -e
        Update-PathFromRegistry
        $pythonExe = Resolve-Python
    }

    # Metodo 2 (fallback): download dell'installer ufficiale da python.org.
    if (-not $pythonExe) {
        $versionePy = "3.13.1"
        $urlPy = "https://www.python.org/ftp/python/$versionePy/python-$versionePy-amd64.exe"
        $installerPy = Join-Path $env:TEMP "python-$versionePy-amd64.exe"
        Write-Host "Download dell'installer ufficiale di Python $versionePy in corso..." -ForegroundColor Cyan
        Write-Host "(Il download richiede circa 25 MB e qualche istante.)" -ForegroundColor Yellow
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $urlPy -OutFile $installerPy -UseBasicParsing
        } catch {
            Write-Host "ERRORE: impossibile scaricare Python. Verifica la connessione a internet." -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
            Pause
            Exit 1
        }
        Write-Host "Installazione di Python in corso (modalità silenziosa)..." -ForegroundColor Cyan
        Start-Process -FilePath $installerPy -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=1","Include_pip=1","Include_launcher=1" -Wait
        Remove-Item $installerPy -ErrorAction SilentlyContinue
        Update-PathFromRegistry
        $pythonExe = Resolve-Python
    }

    if (-not $pythonExe) {
        Write-Host "ERRORE: installazione automatica di Python non riuscita." -ForegroundColor Red
        Write-Host "Installa manualmente Python 3.13 (minimo $($PYTHON_VERSIONE_MINIMA)) da https://www.python.org/downloads/ e riesegui questo script." -ForegroundColor Yellow
        Pause
        Exit 1
    }
    Write-Host "Python installato correttamente." -ForegroundColor Green
}

# Mostra versione Python rilevata
$pyVersion = & $pythonExe --version
Write-Host "Rilevato: $pyVersion" -ForegroundColor Green

# 2. Creazione dell'ambiente virtuale venv
Write-Host "`n[1/6] Creazione dell'ambiente virtuale (venv) in corso..." -ForegroundColor Cyan
& $pythonExe -m venv venv
if (-not $?) {
    Write-Host "ERRORE: Impossibile creare l'ambiente virtuale 'venv'." -ForegroundColor Red
    Pause
    Exit 1
}
Write-Host "Ambiente virtuale creato con successo!" -ForegroundColor Green

# 3. Aggiornamento pip interno
Write-Host "`n[2/6] Aggiornamento di pip nell'ambiente virtuale..." -ForegroundColor Cyan
.\venv\Scripts\python -m pip install --upgrade pip

# 4. Rilevamento GPU NVIDIA per scegliere la build corretta di PyTorch
Write-Host "`n[3/6] Rilevamento hardware (GPU NVIDIA)..." -ForegroundColor Cyan
$gpuNvidia = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # La presenza di nvidia-smi indica driver NVIDIA installati
    $gpuNvidia = $true
} else {
    # In alternativa controlla le schede video tramite WMI/CIM
    try {
        $schedeVideo = Get-CimInstance Win32_VideoController -ErrorAction Stop
        if ($schedeVideo | Where-Object { $_.Name -match 'NVIDIA' }) {
            $gpuNvidia = $true
        }
    } catch {}
}

if ($gpuNvidia) {
    $indiceTorch = "https://download.pytorch.org/whl/cu124"
    Write-Host "GPU NVIDIA rilevata: verrà installato PyTorch con supporto CUDA (cu124)." -ForegroundColor Green
} else {
    $indiceTorch = "https://download.pytorch.org/whl/cpu"
    Write-Host "Nessuna GPU NVIDIA rilevata: verrà installato PyTorch in versione CPU (download più leggero)." -ForegroundColor Yellow
}

# 5. Installazione di PyTorch dalla build corretta (CUDA o CPU)
Write-Host "`n[4/6] Installazione di PyTorch (torch, torchvision)..." -ForegroundColor Cyan
Write-Host "Questa operazione potrebbe richiedere diversi minuti a seconda della connessione internet..." -ForegroundColor Yellow
.\venv\Scripts\python -m pip install torch torchvision --index-url $indiceTorch
if (-not $?) {
    Write-Host "ERRORE: Installazione di PyTorch fallita." -ForegroundColor Red
    Pause
    Exit 1
}

# 6. Installazione degli altri pacchetti base leggendoli da requirements.txt
#    (torch/torchvision sono gia' stati installati allo step precedente dalla
#    index CUDA/CPU corretta: pip li vedra' gia' soddisfatti e non li reinstallera'
#    da PyPI. Aggiungere una dipendenza a requirements.txt e' quindi sufficiente,
#    senza dover modificare questo script.)
Write-Host "`n[5/6] Installazione dei pacchetti base da requirements.txt (Whisper, Streamlit, OpenCV, ChromaDB)..." -ForegroundColor Cyan
.\venv\Scripts\python -m pip install -r requirements.txt
if (-not $?) {
    Write-Host "ERRORE: Installazione delle dipendenze di base fallita." -ForegroundColor Red
    Pause
    Exit 1
}

# 5. Installazione modelli local-only no-deps
Write-Host "`n[6/6] Installazione di facenet-pytorch (--no-deps, per evitare NumPy < 2.0)..." -ForegroundColor Cyan
.\venv\Scripts\python -m pip install facenet-pytorch --no-deps
if (-not $?) {
    Write-Host "ERRORE: Installazione del pacchetto FaceNet fallita." -ForegroundColor Red
    Pause
    Exit 1
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " INSTALLAZIONE COMPLETATA CON SUCCESSO!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Per avviare l'applicazione puoi ora utilizzare:" -ForegroundColor Yellow
Write-Host " - Il file eseguibile: doppio clic su scripts\windows\run.bat" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Green

# 7. Download automatico dei modelli Qwen (~2.1 GB da Hugging Face) e di
#    llama-server (release ufficiale llama.cpp, build CPU). Idempotente: i file
#    gia' presenti non vengono riscaricati. Non bloccante: se il download
#    fallisce l'app parte comunque e la coda segnera' gli embedding come
#    falliti, recuperabili con "Riprova falliti" dopo aver rieseguito lo script.
Write-Host "`n[Extra] Download modelli Qwen + llama-server (se mancanti)..." -ForegroundColor Cyan
$dirQwen = Join-Path (Get-Location) "models\qwen"
New-Item -ItemType Directory -Force -Path $dirQwen | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest e' lentissimo con la barra attiva

# Scarica su file .part e rinomina a fine download: un download interrotto non
# lascia mai un file parziale col nome definitivo (che verrebbe saltato al retry).
function Get-FileSePossibile {
    param([string]$url, [string]$dest)
    if (Test-Path $dest) { Write-Host "  gia' presente: $(Split-Path -Leaf $dest)" -ForegroundColor Green; return $true }
    Write-Host "  download: $(Split-Path -Leaf $dest)" -ForegroundColor Cyan
    try {
        # curl.exe (incluso in Windows 10+) e' molto piu' veloce e riprende i download interrotti
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            curl.exe -L --fail --retry 3 -C - -o "$dest.part" $url
            if ($LASTEXITCODE -ne 0) { throw "curl exit $LASTEXITCODE" }
        } else {
            Invoke-WebRequest -Uri $url -OutFile "$dest.part" -UseBasicParsing
        }
        # La rinomina puo' fallire con "file in uso da un altro processo": Windows
        # Defender scansiona i file grandi appena scritti e li blocca per qualche
        # secondo. Si riprova fino a 5 volte prima di arrendersi.
        foreach ($tentativo in 1..5) {
            try {
                Move-Item -Force "$dest.part" $dest -ErrorAction Stop
                break
            } catch {
                if ($tentativo -eq 5) { throw }
                Start-Sleep -Seconds 3
            }
        }
        return $true
    } catch {
        Write-Host "  FALLITO: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

$urlHf = "https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF/resolve/main"

# SHA-256 ufficiali (metadati LFS del repo Hugging Face). La verifica e'
# obbligatoria: un GGUF corrotto NON fa fallire llama-server, produce embedding
# NaN silenziosi (successo davvero: ricerca rotta senza alcun errore visibile).
$shaAttesi = @{
    "Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf"     = "3f2f9023f15d5f3f084034eb5f14cc04a8e8d89b1f262354db9cf63c50308206"
    "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf" = "3f89a7768ffa6606935319f71bf56bb71871249ba549bf1080a0caea7a088613"
}

$okModelli = $true
foreach ($nome in @("Qwen.Qwen3-VL-Embedding-2B.Q5_K_M.gguf", "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf")) {
    $dest = Join-Path $dirQwen $nome
    if (-not (Get-FileSePossibile "$urlHf/$nome" $dest)) { $okModelli = $false; continue }
    Write-Host "  verifica SHA-256: $nome" -ForegroundColor Cyan
    $shaFile = (Get-FileHash -Algorithm SHA256 $dest).Hash.ToLower()
    if ($shaFile -ne $shaAttesi[$nome]) {
        Remove-Item $dest -ErrorAction SilentlyContinue
        Write-Host "  CORROTTO (hash non corrispondente): $nome eliminato. Rilancia lo script per riscaricarlo." -ForegroundColor Red
        $okModelli = $false
    }
}

# llama-server: release pinnata di llama.cpp (build CPU: l'app lo lancia con -ngl 0).
# ponytail: versione fissa b10016, da alzare a mano se una futura quantizzazione la richiede.
$okServer = $true
if (-not (Test-Path (Join-Path $dirQwen "llama-server.exe"))) {
    $tagLlama = "b10016"
    $zipLlama = Join-Path $env:TEMP "llama-$tagLlama-bin-win-cpu-x64.zip"
    $urlLlama = "https://github.com/ggml-org/llama.cpp/releases/download/$tagLlama/llama-$tagLlama-bin-win-cpu-x64.zip"
    if (Get-FileSePossibile $urlLlama $zipLlama) {
        $dirEstrazione = Join-Path $env:TEMP "llama-estratto"
        if (Test-Path $dirEstrazione) { Remove-Item -Recurse -Force $dirEstrazione }
        Expand-Archive -Path $zipLlama -DestinationPath $dirEstrazione -Force
        # Il layout interno dello zip e' cambiato tra le release: si cerca l'exe
        # ovunque e si copiano exe + DLL dalla sua stessa cartella.
        $exe = Get-ChildItem -Path $dirEstrazione -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if ($exe) {
            Get-ChildItem -Path $exe.DirectoryName -Include "*.exe","*.dll" -Recurse |
                Copy-Item -Destination $dirQwen -Force
            Write-Host "  llama-server.exe installato in models\qwen\" -ForegroundColor Green
        } else {
            Write-Host "  FALLITO: llama-server.exe non trovato nello zip." -ForegroundColor Red
            $okServer = $false
        }
        Remove-Item -Recurse -Force $dirEstrazione -ErrorAction SilentlyContinue
        Remove-Item -Force $zipLlama -ErrorAction SilentlyContinue
    } else { $okServer = $false }
} else {
    Write-Host "  gia' presente: llama-server.exe" -ForegroundColor Green
}

if (-not ($okModelli -and $okServer)) {
    Write-Host "`nATTENZIONE: download dei modelli incompleto. Riesegui questo script con" -ForegroundColor Yellow
    Write-Host "una connessione attiva, oppure copia i file a mano in models\qwen\ (vedi README)." -ForegroundColor Yellow
}

Pause

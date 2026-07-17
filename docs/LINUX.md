# DeepSight su Linux

## Supporto

La release Linux usa come baseline Ubuntu 22.04/24.04 x86-64. CPU è sempre
supportata; CUDA è opzionale per FaceNet e Whisper. Qwen usa il binario CPU di
`llama-server` su Linux. macOS e ARM64 richiedono pacchetti dedicati e non sono
inclusi in questo installer.

## Installazione nuova

Prerequisiti Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip curl tar coreutils
```

Profilo più compatibile:

```bash
git clone -b v2_Ingrassia https://github.com/Fringedend/Quantizzati_DeepSight.git
cd Quantizzati_DeepSight
./scripts/linux/install.sh
./scripts/linux/run.sh
```

Opzioni:

```bash
./scripts/linux/install.sh --cuda
./scripts/linux/install.sh --prefetch-models
./scripts/linux/install.sh --cuda --prefetch-models
```

`--cuda` installa PyTorch CUDA ma Qwen resta CPU. Se la versione CUDA predefinita
non è adatta, si può indicare un indice PyTorch ufficiale:

```bash
DEEPSIGHT_TORCH_INDEX=https://download.pytorch.org/whl/cuXXX \
  ./scripts/linux/install.sh --cuda
```

## Verifica

```bash
./venv/bin/python scripts/check_install.py
./venv/bin/python scripts/smoke_media.py
./venv/bin/python scripts/smoke_app.py
./venv/bin/python src/test_models.py
```

I primi tre comandi sono leggeri: verificano installazione, JPEG/MP4/FFmpeg e
l'avvio HTTP di Streamlit usando database temporanei. `test_models.py` carica
davvero Qwen, FaceNet e Whisper e può richiedere diversi minuti.

## Trasferimento da Windows

Chiudere DeepSight su Windows e copiare almeno:

```text
data/
```

`models/` può essere copiato solo per i file GGUF; il binario Windows
`llama-server.exe` non è utilizzabile su Linux. È più semplice lasciare che
l'installer Linux completi `models/`.

Sul PC Linux:

```bash
./scripts/linux/install.sh
./venv/bin/python scripts/migrate_paths.py --dry-run
./venv/bin/python scripts/migrate_paths.py --apply
./venv/bin/python scripts/rebuild_vector_index.py
./scripts/linux/run.sh
```

La migrazione crea un backup timestampato prima di scrivere. L'app esegue anche
una conversione automatica idempotente al primo avvio, ma il comando manuale
permette di controllare l'anteprima.

Non è necessario copiare `chroma_db/`: SQLite conserva gli embedding e lo script
ricostruisce/riallinea gli indici.

## Backup

Con l'app chiusa, salvare:

```text
data/
```

Per un ripristino completo offline salvare anche:

```text
models/
```

`venv/`, `chroma_db/` e `__pycache__/` sono ricreabili.

## Problemi comuni

`Permission denied`: verificare di avere clonato la versione con gli script
eseguibili; come fallback usare `bash scripts/linux/install.sh`.

`venv non è Linux`: non copiare un virtual environment Windows. Rimuoverlo o
spostarlo manualmente e rilanciare l'installatore.

`llama-server non è avviabile`: la diagnostica controlla eseguibile e librerie;
rieseguire l'installer su Ubuntu x86-64.

`originale non trovato` dopo un trasferimento: eseguire prima `migrate_paths.py
--dry-run`, poi `--apply`, e verificare che `data/archive/` sia stato copiato.

La funzione “Apri cartella” richiede normalmente `xdg-open` (`xdg-utils`), ma la
mancanza di questa utility non impedisce ricerca, galleria o download.

## Pacchetto di rilascio

Dopo commit e test, creare il pacchetto Linux da Git: in questo modo entrano solo
i file versionati, gli `.sh` mantengono il bit eseguibile e restano esclusi
automaticamente `venv/`, `data/`, `models/`, `chroma_db/` e `__pycache__/`.

```bash
mkdir -p dist
git archive --format=tar.gz --prefix=DeepSight/ \
  --output=dist/DeepSight-linux-x86_64.tar.gz HEAD
```

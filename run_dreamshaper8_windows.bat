@echo off
setlocal

cd /d "%~dp0"

set "EMAIL=luispessoa18@gmail.com"
set "MODEL_REPO=Lykon/dreamshaper-8"
set "MODEL_DIR=%CD%\models\dreamshaper-8"
set "VENV_DIR=%CD%\.venv"

echo.
echo HidraImg Worker - DreamShaper 8
echo Email: %EMAIL%
echo Modelo: %MODEL_REPO%
echo Pasta local: %MODEL_DIR%
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python nao encontrado no PATH.
    echo Instale Python 3.10+ e tente novamente.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Criando ambiente virtual...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error

echo Atualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Instalando PyTorch CUDA e dependencias...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :error

python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Baixando modelo antes de iniciar o worker...
python download_model.py --model "%MODEL_REPO%" --output "%MODEL_DIR%"
if errorlevel 1 goto :error

set "HIDRACHAT_WORKER_EMAIL=%EMAIL%"
set "HIDRACHAT_MODEL_ID=%MODEL_DIR%"
set "HIDRACHAT_DEVICE=cuda"
set "HIDRACHAT_TORCH_DTYPE=auto"
set "HIDRACHAT_WORKER_NAME=image-worker-desktop"
set "HIDRACHAT_REGION=desktop"
set "HIDRACHAT_LOCAL_FILES_ONLY=1"
set "HIDRACHAT_PRELOAD_MODEL=1"
set "HIDRACHAT_WARMUP_MODEL=1"

echo.
echo Iniciando worker. O modelo sera carregado na GPU antes de registrar.
python worker.py
if errorlevel 1 goto :error

goto :done

:error
echo.
echo Falhou. Veja a mensagem acima.
pause
exit /b 1

:done
endlocal

# HidraImg Worker

Worker de geração de imagens para a rede [HidraChat](https://hidrachat.cloud), usando **stable-diffusion.cpp**.

Você conecta sua GPU (ou CPU) à rede e gera imagens para outros usuários, ganhando **HidraCoins (HC)** por cada job processado.

---

## Como funciona

1. O usuário digita um prompt no HidraChat
2. Um worker de texto melhora o prompt automaticamente
3. **Este worker** recebe o prompt melhorado e gera a imagem com stable-diffusion.cpp
4. A imagem é enviada de volta ao servidor e exibida para o usuário

---

## Pré-requisitos

- Python 3.10+
- Git + CMake
- Conta no [HidraChat](https://hidrachat.cloud) (grátis)
- Um modelo Stable Diffusion (`.safetensors` ou `.gguf`)

---

## Instalação

### 1. Clone este repositório

```bash
git clone https://github.com/Luispessoa18/hidrachat-image-worker
cd hidrachat-image-worker
```

### 2. Compile o stable-diffusion.cpp

Clone e compile dentro da pasta do worker:

```bash
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
mkdir build && cd build
```

Escolha o comando de build para sua plataforma:

---

#### Linux — CPU

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

#### Linux — NVIDIA CUDA

```bash
cmake .. -DSD_CUBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

#### Linux — AMD ROCm

```bash
cmake .. -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

#### Linux — Vulkan (AMD/Intel/qualquer GPU)

```bash
cmake .. -DSD_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

---

#### macOS — CPU / Apple Silicon (Metal)

```bash
cmake .. -DSD_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)
```

---

#### Windows — CPU (MinGW / MSYS2)

Abra o terminal MSYS2 MinGW64:

```bash
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j4
```

#### Windows — NVIDIA CUDA (Visual Studio)

Abra o **Developer Command Prompt for VS**:

```cmd
cmake .. -DSD_CUBLAS=ON
cmake --build . --config Release
```

#### Windows — Vulkan (AMD/Intel)

```cmd
cmake .. -DSD_VULKAN=ON
cmake --build . --config Release
```

---

Após compilar, o binário estará em:
- Linux/macOS: `stable-diffusion.cpp/build/bin/sd`
- Windows: `stable-diffusion.cpp\build\bin\Release\sd.exe`

### 3. Baixe um modelo

Crie a pasta `models/` e baixe um modelo compatível:

```bash
mkdir models
pip install huggingface_hub

# SD 1.5 — leve, rápido, ~2 GB RAM
huggingface-cli download runwayml/stable-diffusion-v1-5 \
  v1-5-pruned-emaonly.safetensors --local-dir models/

# SDXL — melhor qualidade, ~6 GB RAM
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
  sd_xl_base_1.0.safetensors --local-dir models/

# FLUX.1 Schnell GGUF — alta qualidade, quantizado
huggingface-cli download leejet/FLUX.1-schnell-gguf \
  flux1-schnell-q4_0.gguf --local-dir models/
```

> **Dica:** Para máquinas com pouca RAM, use SD 1.5 (`512×512`).
> Para GPUs modernas, FLUX.1 dá resultados muito melhores.

---

## Rodando o worker

### Linux / macOS

```bash
# CPU (mínimo)
HIDRACHAT_WORKER_EMAIL=seu@email.com python worker.py

# GPU NVIDIA
HIDRACHAT_WORKER_EMAIL=seu@email.com \
HIDRACHAT_N_GPU_LAYERS=35 \
python worker.py
```

### Windows (PowerShell)

```powershell
$env:HIDRACHAT_WORKER_EMAIL = "seu@email.com"
$env:HIDRACHAT_N_GPU_LAYERS = "35"   # remova se for CPU
python worker.py
```

### Windows (cmd)

```cmd
set HIDRACHAT_WORKER_EMAIL=seu@email.com
set HIDRACHAT_N_GPU_LAYERS=35
python worker.py
```

---

O worker vai perguntar qual modelo usar na primeira vez. Na próxima execução, defina `HIDRACHAT_MODEL_PATH` para pular a pergunta:

```bash
HIDRACHAT_MODEL_PATH=models/v1-5-pruned-emaonly.safetensors \
HIDRACHAT_WORKER_EMAIL=seu@email.com \
python worker.py
```

---

## Variáveis de ambiente

| Variável | Default | Descrição |
|---|---|---|
| `HIDRACHAT_ROOT_URL` | `https://hidrachat.cloud` | URL do servidor HidraChat |
| `HIDRACHAT_WORKER_NAME` | `image-worker` | Nome deste worker no painel |
| `HIDRACHAT_WORKER_EMAIL` | — | Email da sua conta HidraChat (**obrigatório**) |
| `HIDRACHAT_SD_BIN` | `sd` | Caminho para o binário `sd` compilado |
| `HIDRACHAT_MODEL_PATH` | — | Caminho direto para o modelo (pula o menu) |
| `HIDRACHAT_MODELS_DIR` | `./models` | Pasta onde procurar modelos |
| `HIDRACHAT_N_GPU_LAYERS` | `0` | Layers na GPU — `0` = CPU puro |
| `HIDRACHAT_POLL_SECONDS` | `3` | Intervalo de polling em segundos |
| `HIDRACHAT_REGION` | `local` | Região do worker |

---

## Rodando em background

### Linux (systemd)

Crie `/etc/systemd/system/hidraimg.service`:

```ini
[Unit]
Description=HidraImg Worker
After=network.target

[Service]
User=SEU_USUARIO
WorkingDirectory=/caminho/para/hidrachat-image-worker
Environment=HIDRACHAT_WORKER_EMAIL=seu@email.com
Environment=HIDRACHAT_MODEL_PATH=models/v1-5-pruned-emaonly.safetensors
Environment=HIDRACHAT_N_GPU_LAYERS=35
ExecStart=/usr/bin/python3 worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable hidraimg
sudo systemctl start hidraimg
sudo journalctl -u hidraimg -f
```

### Linux / macOS (screen)

```bash
screen -S hidraimg
HIDRACHAT_WORKER_EMAIL=seu@email.com python worker.py
# Ctrl+A, D para desanexar
screen -r hidraimg  # para voltar
```

### Windows (rodando em background com start)

```cmd
start /B python worker.py > hidraimg.log 2>&1
```

---

## Requisitos mínimos

| Configuração | RAM | VRAM | Modelo recomendado |
|---|---|---|---|
| CPU básico | 8 GB | — | SD 1.5 `512×512` |
| CPU robusto | 16 GB | — | SDXL `768×768` |
| GPU 4 GB VRAM | 8 GB | 4 GB | SD 1.5 com GPU |
| GPU 8 GB VRAM | 16 GB | 8 GB | SDXL com GPU |
| GPU 16 GB+ VRAM | 16 GB | 16 GB | FLUX.1 com GPU |

---

## Problemas comuns

**`sd: command not found`**
→ Defina `HIDRACHAT_SD_BIN` com o caminho completo para o binário compilado.

**`Nenhum modelo encontrado`**
→ Crie a pasta `models/` e coloque um `.safetensors` ou `.gguf` nela.

**Geração muito lenta**
→ Use GPU com `HIDRACHAT_N_GPU_LAYERS=35` (ou mais, dependendo da VRAM).

**CUDA out of memory**
→ Reduza `HIDRACHAT_N_GPU_LAYERS` ou use um modelo menor/quantizado.

---

Feito com ❤️ — [HidraChat](https://hidrachat.cloud)

# HidraImg Worker

Worker de geracao de imagens para a rede [HidraChat](https://hidrachat.cloud), usando **HuggingFace Diffusers + PyTorch**.

Voce conecta sua GPU NVIDIA, Colab ou CPU a rede e gera imagens para outros usuarios, ganhando **HidraCoins (HC)** por cada job processado.

---

## Como funciona

1. O usuario digita um prompt no HidraChat
2. Um worker de texto melhora o prompt automaticamente
3. Este worker recebe o prompt melhorado e gera a imagem com Diffusers/PyTorch
4. A imagem e enviada de volta ao servidor e exibida para o usuario

---

## Pre-requisitos

- Python 3.10+
- Conta no [HidraChat](https://hidrachat.cloud)
- PyTorch + Diffusers
- Um modelo Diffusers local ou um repo Hugging Face, como `stabilityai/stable-diffusion-xl-base-1.0`

---

## Instalacao

### 1. Clone este repositorio

```bash
git clone https://github.com/Luispessoa18/hidrachat-image-worker
cd hidrachat-image-worker
```

### 2. Instale PyTorch e Diffusers

Linux/Colab com CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install xformers
```

CPU ou ambiente sem NVIDIA:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 3. Escolha um modelo

Use um repo Hugging Face direto:

```bash
export HIDRACHAT_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
```

Ou baixe um modelo local no formato Diffusers:

```bash
pip install huggingface_hub
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/
```

O worker tambem aceita arquivos `.safetensors`/`.ckpt` via Diffusers `from_single_file`, mas pasta Diffusers costuma ser o caminho mais simples.

---

## Rodando o worker

### Google Colab

Abra o notebook [HidraImg_Colab.ipynb](HidraImg_Colab.ipynb) no Colab, ou veja [COLAB.md](COLAB.md). O caminho direto e:

```python
%env HIDRACHAT_WORKER_EMAIL=luispessoa18@gmail.com
%env HIDRACHAT_MODEL_ID=runwayml/stable-diffusion-v1-5
%env HIDRACHAT_DEVICE=cuda
!python colab_worker.py
```

### Linux / macOS / Colab

```bash
HIDRACHAT_WORKER_EMAIL=seu@email.com \
HIDRACHAT_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 \
python worker.py
```

Para forcar CUDA:

```bash
HIDRACHAT_WORKER_EMAIL=seu@email.com \
HIDRACHAT_DEVICE=cuda \
HIDRACHAT_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 \
python worker.py
```

### Windows PowerShell

```powershell
$env:HIDRACHAT_WORKER_EMAIL = "seu@email.com"
$env:HIDRACHAT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
python worker.py
```

Se houver modelos locais em `models/`, o worker pergunta qual usar. Para rodar sem pergunta, defina `HIDRACHAT_MODEL_ID`.

---

## Variaveis de ambiente

| Variavel | Default | Descricao |
|---|---|---|
| `HIDRACHAT_ROOT_URL` | `https://hidrachat.cloud` | URL do servidor HidraChat |
| `HIDRACHAT_WORKER_NAME` | `image-worker` | Nome deste worker no painel |
| `HIDRACHAT_WORKER_EMAIL` | - | Email da sua conta HidraChat |
| `HIDRACHAT_MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | Repo Hugging Face, pasta Diffusers local ou arquivo `.safetensors`/`.ckpt` |
| `HIDRACHAT_MODEL_PATH` | - | Alias legado para `HIDRACHAT_MODEL_ID` |
| `HIDRACHAT_MODELS_DIR` | `./models` | Pasta onde procurar modelos locais |
| `HIDRACHAT_DEVICE` | `auto` | `auto`, `cuda` ou `cpu` |
| `HIDRACHAT_TORCH_DTYPE` | `auto` | `auto`, `float16`, `bfloat16` ou `float32` |
| `HIDRACHAT_POLL_SECONDS` | `3` | Intervalo de polling em segundos |
| `HIDRACHAT_REGION` | `local` | Regiao do worker |

---

## Systemd

Exemplo `/etc/systemd/system/hidraimg.service`:

```ini
[Unit]
Description=HidraImg Worker
After=network.target

[Service]
User=SEU_USUARIO
WorkingDirectory=/caminho/para/hidrachat-image-worker
Environment=HIDRACHAT_WORKER_EMAIL=seu@email.com
Environment=HIDRACHAT_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
Environment=HIDRACHAT_DEVICE=cuda
ExecStart=/usr/bin/python3 worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Requisitos minimos

| Configuracao | RAM | VRAM | Modelo recomendado |
|---|---|---|---|
| CPU basico | 8 GB | - | SD 1.5 `512x512` |
| GPU 4 GB VRAM | 8 GB | 4 GB | SD 1.5 |
| GPU 8 GB VRAM | 16 GB | 8 GB | SDXL |
| GPU 16 GB+ VRAM | 16 GB | 16 GB | SDXL ou modelos maiores |

---

## Problemas comuns

**`No module named diffusers`**
Instale as dependencias com `pip install -r requirements.txt`.

**Erro ao baixar modelo Hugging Face**
Verifique internet/token do Hugging Face ou use `HIDRACHAT_MODEL_ID` apontando para uma pasta local ja baixada.

**Geracao muito lenta**
Use GPU NVIDIA com `HIDRACHAT_DEVICE=cuda`.

**`CUDA out of memory`**
Reduza largura/altura/steps, use modelo menor ou rode com `HIDRACHAT_TORCH_DTYPE=float16`.

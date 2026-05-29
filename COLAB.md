# Rodar no Google Colab

Use runtime com GPU: `Runtime > Change runtime type > T4/A100 GPU`.

## Opcao 1: por variaveis de ambiente

```python
!git clone https://github.com/Luispessoa18/hidrachat-image-worker
%cd hidrachat-image-worker
```

```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q -r requirements.txt
!pip install -q xformers
```

```python
%env HIDRACHAT_WORKER_EMAIL=luispessoa18@gmail.com
%env HIDRACHAT_MODEL_ID=runwayml/stable-diffusion-v1-5
%env HIDRACHAT_DEVICE=cuda
%env HIDRACHAT_WORKER_NAME=image-worker-colab
```

```python
!python colab_worker.py
```

## Opcao 2: por import no notebook

```python
from colab_worker import start

start(
    email="luispessoa18@gmail.com",
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda",
)
```

## Baixar modelo antes de iniciar

```python
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LOCAL_MODEL_DIR = "/content/hidrachat-image-worker/models/sd15"

!huggingface-cli download {MODEL_ID} --local-dir {LOCAL_MODEL_DIR} --local-dir-use-symlinks False
%env HIDRACHAT_MODEL_ID=/content/hidrachat-image-worker/models/sd15
```

## Modelos leves para testar

Para testar mais rapido em T4, use SD 1.5:

```python
%env HIDRACHAT_MODEL_ID=runwayml/stable-diffusion-v1-5
```

Para SDXL:

```python
%env HIDRACHAT_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
```

Se o modelo exigir aceite/licenca no Hugging Face, faca login antes:

```python
from huggingface_hub import login
login()
```

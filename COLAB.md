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
%env HIDRACHAT_MODEL_ID=/content/hidrachat-image-worker/models/sd15
%env HIDRACHAT_DEVICE=cuda
%env HIDRACHAT_WORKER_NAME=image-worker-colab
%env HIDRACHAT_LOCAL_FILES_ONLY=1
%env HIDRACHAT_PRELOAD_MODEL=1
%env HIDRACHAT_WARMUP_MODEL=1
```

```python
!python download_model.py --model runwayml/stable-diffusion-v1-5 --output /content/hidrachat-image-worker/models/sd15
```

```python
!python colab_worker.py
```

## Opcao 2: por import no notebook

```python
from colab_worker import start

start(
    email="luispessoa18@gmail.com",
    model_id="/content/hidrachat-image-worker/models/sd15",
    device="cuda",
)
```

## Baixar modelo antes de iniciar

```python
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LOCAL_MODEL_DIR = "/content/hidrachat-image-worker/models/sd15"

!python download_model.py --model {MODEL_ID} --output {LOCAL_MODEL_DIR}
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

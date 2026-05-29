"""
Entrada simples para rodar o HidraImg Worker no Google Colab.

Uso por variaveis de ambiente:
  HIDRACHAT_WORKER_EMAIL=voce@email.com \
  HIDRACHAT_MODEL_ID=/content/hidrachat-image-worker/models/sd15 \
  python colab_worker.py

Uso por import no notebook:
  from colab_worker import start
  start(email="voce@email.com", model_id="/content/hidrachat-image-worker/models/sd15")
"""

from __future__ import annotations

import argparse
import os
from typing import Any


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def configure(
    *,
    email: str | None = None,
    model_id: str | None = None,
    device: str = "cuda",
    dtype: str = "auto",
    root_url: str = "https://hidrachat.cloud",
    worker_name: str = "image-worker-colab",
    poll_seconds: float = 3,
    region: str = "colab",
    local_files_only: bool = True,
    preload_model: bool = True,
) -> None:
    if email:
        os.environ["HIDRACHAT_WORKER_EMAIL"] = email
    if model_id:
        os.environ["HIDRACHAT_MODEL_ID"] = model_id

    os.environ.setdefault("HIDRACHAT_MODEL_ID", DEFAULT_MODEL_ID)
    os.environ["HIDRACHAT_DEVICE"] = device
    os.environ["HIDRACHAT_TORCH_DTYPE"] = dtype
    os.environ["HIDRACHAT_ROOT_URL"] = root_url
    os.environ["HIDRACHAT_WORKER_NAME"] = worker_name
    os.environ["HIDRACHAT_POLL_SECONDS"] = str(poll_seconds)
    os.environ["HIDRACHAT_REGION"] = region
    os.environ["HIDRACHAT_LOCAL_FILES_ONLY"] = "1" if local_files_only else "0"
    os.environ["HIDRACHAT_PRELOAD_MODEL"] = "1" if preload_model else "0"


def start(**kwargs: Any) -> None:
    configure(**kwargs)
    import worker

    worker.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start HidraImg Worker on Colab")
    parser.add_argument("--email", default=os.getenv("HIDRACHAT_WORKER_EMAIL", ""))
    parser.add_argument("--model", default=os.getenv("HIDRACHAT_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--device", default=os.getenv("HIDRACHAT_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.getenv("HIDRACHAT_TORCH_DTYPE", "auto"))
    parser.add_argument("--root-url", default=os.getenv("HIDRACHAT_ROOT_URL", "https://hidrachat.cloud"))
    parser.add_argument("--name", default=os.getenv("HIDRACHAT_WORKER_NAME", "image-worker-colab"))
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("HIDRACHAT_POLL_SECONDS", "3")))
    parser.add_argument("--region", default=os.getenv("HIDRACHAT_REGION", "colab"))
    parser.add_argument("--online-model", action="store_true", help="Permite baixar/cachear modelo durante o start")
    parser.add_argument("--no-preload", action="store_true", help="Registra antes de carregar o modelo")
    args = parser.parse_args()

    if not args.email:
        raise SystemExit("Defina HIDRACHAT_WORKER_EMAIL ou passe --email.")

    start(
        email=args.email,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        root_url=args.root_url,
        worker_name=args.name,
        poll_seconds=args.poll_seconds,
        region=args.region,
        local_files_only=not args.online_model,
        preload_model=not args.no_preload,
    )


if __name__ == "__main__":
    main()

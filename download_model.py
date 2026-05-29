"""
Baixa apenas os arquivos necessarios de um modelo Diffusers.

Isso evita o comportamento do `huggingface-cli download`, que pode baixar pesos
extras de PyTorch/Flax/ONNX e passar de dezenas de GB em alguns repositorios.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "feature_extractor/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/pytorch_model.bin",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "text_encoder_2/pytorch_model.bin",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.bin",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.bin",
]

IGNORE_PATTERNS = [
    "*.ckpt",
    "*.msgpack",
    "*.onnx",
    "*.pb",
    "*.h5",
    "*.ot",
    "flax_model.*",
    "tf_model.*",
    "onnx/*",
    "openvino/*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download minimal Diffusers snapshot")
    parser.add_argument("--model", required=True, help="Repo Hugging Face, ex: runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output", required=True, help="Pasta local de destino")
    parser.add_argument("--revision", default=None, help="Revision opcional, ex: fp16")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model,
        local_dir=str(out),
        revision=args.revision,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        local_dir_use_symlinks=False,
    )

    print(f"Modelo baixado em: {out.resolve()}")


if __name__ == "__main__":
    main()

"""
HidraImg Worker — gera imagens com HuggingFace Diffusers/PyTorch
Registra como worker_type=image e processa apenas jobs image_generation.

Variáveis de ambiente:
  HIDRACHAT_ROOT_URL      URL do servidor (default: https://hidrachat.cloud)
  HIDRACHAT_WORKER_NAME   Nome deste worker
  HIDRACHAT_WORKER_EMAIL  Email da conta dona do worker
  HIDRACHAT_MODEL_ID      Repo HuggingFace ou pasta Diffusers local
  HIDRACHAT_MODEL_PATH    Alias legado para HIDRACHAT_MODEL_ID
  HIDRACHAT_MODELS_DIR    Pasta onde procurar modelos Diffusers locais
  HIDRACHAT_OUTPUT_DIR    Pasta para salvar imagens geradas (default: ./output)
  HIDRACHAT_POLL_SECONDS  Intervalo de polling (default: 3)
  HIDRACHAT_DEVICE        cuda/cpu/auto (default: auto)
  HIDRACHAT_TORCH_DTYPE   auto/float16/bfloat16/float32 (default: auto)
  HIDRACHAT_LOCAL_FILES_ONLY  1 para usar apenas modelo local/cache
  HIDRACHAT_PRELOAD_MODEL     1 para carregar modelo antes de registrar
  HIDRACHAT_WARMUP_MODEL      1 para rodar uma geracao curta antes de registrar
  HIDRACHAT_REGION        Região do worker (default: local)
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("HIDRACHAT_OUTPUT_DIR", str(ROOT_DIR / "output")))
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class Config:
    root_url:     str   = os.getenv("HIDRACHAT_ROOT_URL",    "https://hidrachat.cloud")
    name:         str   = os.getenv("HIDRACHAT_WORKER_NAME", "image-worker")
    owner_email:  str   = os.getenv("HIDRACHAT_WORKER_EMAIL", "")
    model_id:     str   = os.getenv(
        "HIDRACHAT_MODEL_ID",
        os.getenv("HIDRACHAT_MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0"),
    )
    region:       str   = os.getenv("HIDRACHAT_REGION",      "local")
    poll_seconds: float = float(os.getenv("HIDRACHAT_POLL_SECONDS", "3"))
    device:       str   = os.getenv("HIDRACHAT_DEVICE",      "auto")
    torch_dtype:  str   = os.getenv("HIDRACHAT_TORCH_DTYPE", "auto")
    local_files_only: bool = os.getenv("HIDRACHAT_LOCAL_FILES_ONLY", "0") == "1"
    preload_model: bool = os.getenv("HIDRACHAT_PRELOAD_MODEL", "0") == "1"
    warmup_model: bool = os.getenv("HIDRACHAT_WARMUP_MODEL", "0") == "1"
    ram_gb:       float = float(os.getenv("HIDRACHAT_RAM_GB", "8"))


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "HidraImageWorker/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ─── Hardware detection ───────────────────────────────────────────────────────

def detect_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        pass
    return float(os.getenv("HIDRACHAT_RAM_GB", "8"))


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def detect_gpu(device: str) -> str:
    if device != "cuda":
        return "CPU"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "CUDA GPU"


# ─── Model discovery ─────────────────────────────────────────────────────────

def find_models() -> list[Path]:
    models_dir = Path(os.getenv("HIDRACHAT_MODELS_DIR", str(ROOT_DIR / "models")))
    if not models_dir.exists():
        return []
    exts = (".safetensors", ".ckpt")
    found = [p for p in models_dir.rglob("*") if p.suffix.lower() in exts]
    found.extend(p.parent for p in models_dir.rglob("model_index.json"))
    return sorted(set(found), key=lambda p: str(p).lower())


def choose_model(default_model_id: str) -> str:
    env = os.getenv("HIDRACHAT_MODEL_ID") or os.getenv("HIDRACHAT_MODEL_PATH")
    if env:
        return env
    models = find_models()
    if not models:
        return default_model_id
    print("\nModelos encontrados:")
    for i, m in enumerate(models, 1):
        if m.is_dir():
            print(f"  {i}. {m}")
        else:
            mb = m.stat().st_size / (1024 * 1024)
            print(f"  {i}. {m.name} ({mb:.0f} MB)")
    while True:
        choice = input(f"Escolha o modelo [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return str(models[int(choice) - 1])
        print("Opção inválida.")


# ─── Image generation ─────────────────────────────────────────────────────────

STYLE_SUFFIXES: dict[str, str] = {
    "realistic":   "photorealistic, ultra-detailed, 8k, sharp focus",
    "anime":       "anime style, cel shading, vibrant colors, Studio Ghibli",
    "artistic":    "oil painting, artistic, masterpiece, trending on artstation",
    "photographic":"professional photography, DSLR, bokeh, studio lighting",
    "cinematic":   "cinematic, movie still, dramatic lighting, anamorphic lens",
    "digital-art": "digital art, concept art, highly detailed, fantasy",
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, "
    "signature, extra limbs, disfigured, deformed"
)

MIN_PNG_BYTES = 1024
PIPELINE: Any | None = None


def resolve_torch_dtype(dtype_name: str, device: str) -> Any:
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def has_safetensors(model_ref: Path) -> bool:
    if model_ref.is_file():
        return model_ref.suffix.lower() == ".safetensors"
    if model_ref.is_dir():
        return any(model_ref.rglob("*.safetensors"))
    return True


def load_pipeline(cfg: Config) -> Any:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    import torch
    from diffusers import AutoPipelineForText2Image

    cfg.device = resolve_device(cfg.device)
    dtype = resolve_torch_dtype(cfg.torch_dtype, cfg.device)
    model_ref = Path(cfg.model_id)
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": has_safetensors(model_ref),
    }
    if cfg.local_files_only:
        load_kwargs["local_files_only"] = True

    print(f"Carregando Diffusers: {cfg.model_id}")
    if model_ref.exists() and model_ref.is_file():
        pipe = AutoPipelineForText2Image.from_single_file(str(model_ref), **load_kwargs)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            cfg.model_id,
            safety_checker=None,
            requires_safety_checker=False,
            **load_kwargs,
        )

    if cfg.device == "cuda":
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    PIPELINE = pipe
    return PIPELINE


def warmup_pipeline(cfg: Config) -> None:
    pipe = load_pipeline(cfg)
    print("Aquecendo pipeline na GPU antes do registro...")
    try:
        import torch

        generator = torch.Generator(device=cfg.device).manual_seed(1)
    except Exception:
        generator = None

    pipe(
        prompt="warmup",
        negative_prompt=NEGATIVE_PROMPT,
        width=256,
        height=256,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]
    print("Pipeline pronto na GPU.\n")


def truncate_prompt_for_pipeline(pipe: Any, prompt: str) -> str:
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return prompt

    max_length = min(int(getattr(tokenizer, "model_max_length", 77)), 77)
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=False,
    )
    input_ids = encoded.get("input_ids")
    if not input_ids:
        return prompt

    truncated = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    if truncated and truncated != prompt:
        print(f"  prompt truncado para {max_length} tokens CLIP")
    return truncated or prompt


def png_bytes_from_image(image: Any, width: int, height: int) -> bytes:
    if getattr(image, "size", None) != (width, height):
        raise RuntimeError(f"imagem gerada com tamanho inesperado: {getattr(image, 'size', None)}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("saida gerada nao e um PNG valido")
    if len(data) < MIN_PNG_BYTES:
        raise RuntimeError(f"PNG gerado pequeno demais ({len(data)} bytes)")
    return data


def generate_image(cfg: Config, prompt: str, constraints: dict) -> bytes:
    width  = int(constraints.get("width",  512))
    height = int(constraints.get("height", 512))
    steps  = int(constraints.get("steps",  20))
    style  = constraints.get("style", "realistic")
    guidance_scale = float(constraints.get("guidance_scale", 7.5))
    seed = constraints.get("seed")

    suffix = STYLE_SUFFIXES.get(style, "")
    full_prompt = f"{prompt}, {suffix}" if suffix else prompt

    pipe = load_pipeline(cfg)
    full_prompt = truncate_prompt_for_pipeline(pipe, full_prompt)

    print(f"  prompt: {full_prompt[:80]}...")
    print(f"  size: {width}x{height}  steps: {steps}  style: {style}")
    generator = None
    if seed is not None:
        import torch

        generator = torch.Generator(device=cfg.device).manual_seed(int(seed))

    image = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return png_bytes_from_image(image, width, height)


# ─── Register & heartbeat ─────────────────────────────────────────────────────

def register(cfg: Config) -> str:
    ram = detect_ram_gb()
    cfg.device = resolve_device(cfg.device)
    gpu = detect_gpu(cfg.device)
    cfg.ram_gb = ram
    print(f"RAM: {ram} GB  |  Backend: {gpu}")
    res = post_json(
        f"{cfg.root_url}/worker/register",
        {
            "name":             cfg.name,
            "owner_email":      cfg.owner_email,
            "worker_type":      "image",
            "region":           cfg.region,
            "model_name":       Path(cfg.model_id).name,
            "model_size":       "any",
            "ram_gb":           ram,
            "cpu_threads":      os.cpu_count() or 4,
            "gpu":              gpu,
            "tokens_per_second": 0,
            "web_search":       False,
        },
    )
    return res["worker_id"]


def heartbeat(cfg: Config, worker_id: str) -> None:
    post_json(
        f"{cfg.root_url}/worker/heartbeat",
        {"worker_id": worker_id, "ram_available_gb": cfg.ram_gb, "tokens_per_second": 0},
    )


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    if not cfg.owner_email:
        cfg.owner_email = input("Email da conta HidraChat: ").strip().lower()

    cfg.model_id = choose_model(cfg.model_id)
    cfg.device = resolve_device(cfg.device)

    print(f"\nBackend:      Diffusers/PyTorch ({cfg.device})")
    print(f"Modelo:       {cfg.model_id}")
    print(f"Servidor:     {cfg.root_url}\n")

    if cfg.preload_model:
        load_pipeline(cfg)
        print("Modelo carregado antes do registro.\n")
    if cfg.warmup_model:
        warmup_pipeline(cfg)

    worker_id = register(cfg)
    print(f"Worker registrado: {worker_id}\n")

    while True:
        try:
            heartbeat(cfg, worker_id)
            task = get_json(f"{cfg.root_url}/pull-task?worker_id={worker_id}")

            if not task.get("job_id"):
                time.sleep(cfg.poll_seconds)
                continue

            job_id      = task["job_id"]
            prompt      = task.get("prompt", "")
            constraints = {}
            try:
                constraints = json.loads(task.get("constraints_json") or "{}")
            except Exception:
                pass

            print(f"[JOB] {job_id}  ({int(constraints.get('width',512))}×{int(constraints.get('height',512))}  steps={constraints.get('steps',20)})")

            try:
                started  = time.perf_counter()
                img_data = generate_image(cfg, prompt, constraints)
                elapsed  = int((time.perf_counter() - started) * 1000)
                b64      = base64.b64encode(img_data).decode()
                print(f"[DONE] {len(img_data)} bytes ({len(img_data) / 1024:.1f} KB)  {elapsed}ms")
                post_json(
                    f"{cfg.root_url}/job/submit",
                    {
                        "job_id":        job_id,
                        "worker_id":     worker_id,
                        "output":        b64,
                        "input_tokens":  0,
                        "output_tokens": 0,
                        "worker_time_ms": elapsed,
                        "success":       True,
                    },
                    timeout=60,
                )
            except Exception as exc:
                print(f"[FAIL] {exc}")
                post_json(
                    f"{cfg.root_url}/job/submit",
                    {
                        "job_id":    job_id,
                        "worker_id": worker_id,
                        "output":    "",
                        "worker_time_ms": 0,
                        "success":   False,
                        "error":     str(exc),
                    },
                )
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[OFFLINE] {exc}")
            time.sleep(max(5, cfg.poll_seconds))


if __name__ == "__main__":
    main()

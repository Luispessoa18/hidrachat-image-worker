"""
HidraImg Worker — gera imagens com stable-diffusion.cpp
Registra como worker_type=image e processa apenas jobs image_generation.

Variáveis de ambiente:
  HIDRACHAT_ROOT_URL      URL do servidor (default: https://hidrachat.cloud)
  HIDRACHAT_WORKER_NAME   Nome deste worker
  HIDRACHAT_WORKER_EMAIL  Email da conta dona do worker
  HIDRACHAT_SD_BIN        Caminho para o binário 'sd' do stable-diffusion.cpp
  HIDRACHAT_MODELS_DIR    Pasta com os modelos .safetensors/.gguf
  HIDRACHAT_MODEL_PATH    Caminho direto para o modelo (override)
  HIDRACHAT_OUTPUT_DIR    Pasta para salvar imagens geradas (default: ./output)
  HIDRACHAT_POLL_SECONDS  Intervalo de polling (default: 3)
  HIDRACHAT_N_GPU_LAYERS  Layers na GPU (default: 0 = CPU)
  HIDRACHAT_REGION        Região do worker (default: local)
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("HIDRACHAT_OUTPUT_DIR", str(ROOT_DIR / "output")))
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class Config:
    root_url:     str   = os.getenv("HIDRACHAT_ROOT_URL",    "https://hidrachat.cloud")
    name:         str   = os.getenv("HIDRACHAT_WORKER_NAME", "image-worker")
    owner_email:  str   = os.getenv("HIDRACHAT_WORKER_EMAIL", "")
    sd_bin:       str   = os.getenv("HIDRACHAT_SD_BIN",      "sd")
    model_path:   str   = os.getenv("HIDRACHAT_MODEL_PATH",  "")
    region:       str   = os.getenv("HIDRACHAT_REGION",      "local")
    poll_seconds: float = float(os.getenv("HIDRACHAT_POLL_SECONDS", "3"))
    n_gpu_layers: int   = int(os.getenv("HIDRACHAT_N_GPU_LAYERS", "0"))
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


def detect_gpu(n_gpu_layers: int) -> str:
    if n_gpu_layers <= 0:
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
    return "Vulkan/GPU"


# ─── Model discovery ─────────────────────────────────────────────────────────

def find_sd_binary() -> str:
    env = os.getenv("HIDRACHAT_SD_BIN")
    if env:
        return env
    candidates = [
        ROOT_DIR / "stable-diffusion.cpp" / "build" / "bin" / "sd",
        ROOT_DIR / "stable-diffusion.cpp" / "build" / "bin" / "Release" / "sd.exe",
        ROOT_DIR / "sd",
        ROOT_DIR / "sd.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "sd"


def find_models() -> list[Path]:
    models_dir = Path(os.getenv("HIDRACHAT_MODELS_DIR", str(ROOT_DIR / "models")))
    if not models_dir.exists():
        return []
    exts = (".safetensors", ".ckpt", ".gguf")
    return sorted(
        (p for p in models_dir.rglob("*") if p.suffix.lower() in exts),
        key=lambda p: p.name.lower(),
    )


def choose_model() -> Path:
    env = os.getenv("HIDRACHAT_MODEL_PATH")
    if env:
        return Path(env)
    models = find_models()
    if not models:
        raise RuntimeError(
            "Nenhum modelo encontrado em models/. "
            "Coloque um .safetensors/.gguf em models/ ou defina HIDRACHAT_MODEL_PATH."
        )
    print("\nModelos encontrados:")
    for i, m in enumerate(models, 1):
        mb = m.stat().st_size / (1024 * 1024)
        print(f"  {i}. {m.name} ({mb:.0f} MB)")
    while True:
        choice = input(f"Escolha o modelo [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
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


def generate_image(cfg: Config, prompt: str, constraints: dict) -> bytes:
    width  = int(constraints.get("width",  512))
    height = int(constraints.get("height", 512))
    steps  = int(constraints.get("steps",  20))
    style  = constraints.get("style", "realistic")

    suffix = STYLE_SUFFIXES.get(style, "")
    full_prompt = f"{prompt}, {suffix}" if suffix else prompt

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=OUTPUT_DIR) as tmp:
        out_path = tmp.name

    cmd = [
        cfg.sd_bin,
        "-m", cfg.model_path,
        "-p", full_prompt,
        "--negative-prompt", NEGATIVE_PROMPT,
        "-W", str(width),
        "-H", str(height),
        "--steps", str(steps),
        "-o", out_path,
        "--verbose",
    ]
    if cfg.n_gpu_layers > 0:
        cmd.extend(["--n-gpu-layers", str(cfg.n_gpu_layers)])

    print(f"  prompt: {full_prompt[:80]}…")
    print(f"  size: {width}×{height}  steps: {steps}  style: {style}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"sd exited with {result.returncode}")

    img_path = Path(out_path)
    if not img_path.exists() or img_path.stat().st_size == 0:
        # stable-diffusion.cpp appends _001 suffix sometimes
        candidates = list(OUTPUT_DIR.glob(img_path.stem + "*.png"))
        if candidates:
            img_path = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            raise RuntimeError("sd não gerou arquivo de saída")

    data = img_path.read_bytes()
    img_path.unlink(missing_ok=True)
    return data


# ─── Register & heartbeat ─────────────────────────────────────────────────────

def register(cfg: Config) -> str:
    ram = detect_ram_gb()
    gpu = detect_gpu(cfg.n_gpu_layers)
    cfg.ram_gb = ram
    print(f"RAM: {ram} GB  |  Backend: {gpu}")
    res = post_json(
        f"{cfg.root_url}/worker/register",
        {
            "name":             cfg.name,
            "owner_email":      cfg.owner_email,
            "worker_type":      "image",
            "region":           cfg.region,
            "model_name":       Path(cfg.model_path).stem,
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

    cfg.sd_bin    = find_sd_binary()
    model_path    = choose_model()
    cfg.model_path = str(model_path)

    print(f"\nBinário sd:   {cfg.sd_bin}")
    print(f"Modelo:       {cfg.model_path}")
    print(f"Servidor:     {cfg.root_url}\n")

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
                print(f"[DONE] {len(img_data)//1024} KB  {elapsed}ms")
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

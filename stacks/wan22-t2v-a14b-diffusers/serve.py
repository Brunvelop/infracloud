# WAN 2.2 T2V A14B FastAPI server (diffusers)
#
# Modelo: Wan-AI/Wan2.2-T2V-A14B-Diffusers
# Arquitectura: Mixture-of-Experts (MoE) T2V, 14B parámetros activados
#
# Endpoints:
#   GET  /health    → {"status": "ready"} o 503 mientras carga
#   POST /generate  → recibe JSON, devuelve video/mp4
#
# Runtime dependencies (instalados por uv sync):
#   - diffusers (desde git main — la versión PyPI no soporta WAN 2.2 aún)
#   - transformers, accelerate, sentencepiece
#   - imageio[ffmpeg]
#   - fastapi, uvicorn
#   - huggingface_hub[hf_transfer]

import os
import random
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

# ── Env vars (antes de cualquier import de torch) ─────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from huggingface_hub import snapshot_download

from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wan22-serve")

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

# Negative prompt oficial recomendado por Wan-AI
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

DEFAULT_FPS = 16

# ── Estado global ─────────────────────────────────────────────────────────────

_pipe: Optional[WanPipeline] = None
_model_ready: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_vram(tag: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak  = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        log.info(
            f"[VRAM {tag}] allocated={alloc:.2f}GB  peak={peak:.2f}GB  "
            f"free={free/1024**3:.2f}GB  total={total/1024**3:.2f}GB"
        )


# ── Lifespan (descarga + carga del modelo) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe, _model_ready

    log.info(f"Descargando modelo desde HuggingFace: {MODEL_REPO}…")
    _log_vram("pre-load")

    # VAE en float32 (requisito de Wan2.2 según la documentación oficial)
    log.info("Cargando VAE (float32)…")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_REPO,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # Pipeline principal en bfloat16
    log.info("Cargando WanPipeline (bfloat16)…")
    _pipe = WanPipeline.from_pretrained(
        MODEL_REPO,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    _pipe.to("cuda")

    _log_vram("post-load")
    _model_ready = True
    log.info("✅ Pipeline listo — servidor aceptando peticiones.")
    yield

    # Teardown
    log.info("Liberando modelo de VRAM…")
    del _pipe
    torch.cuda.empty_cache()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="WAN 2.2 T2V A14B Server", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None          # usa el default oficial si no se pasa
    width: int = 1280                               # 720p landscape
    height: int = 720
    num_frames: int = 81                            # 81 frames @ 16fps ≈ 5s
    guidance_scale: float = 4.0                    # guidance principal
    guidance_scale_2: float = 3.0                  # guidance secundario (MoE específico)
    num_inference_steps: int = 40
    fps: int = DEFAULT_FPS
    seed: Optional[int] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model": "wan22-t2v-a14b"},
        )
    return {"status": "ready", "model": "wan22-t2v-a14b"}


@app.post("/generate")
@torch.inference_mode()
def generate(req: GenerateRequest):
    """Genera un vídeo MP4 a partir de un prompt de texto.

    Devuelve el vídeo directamente como bytes (Content-Type: video/mp4).
    """
    if not _model_ready:
        return JSONResponse(status_code=503, content={"error": "Modelo no listo"})

    torch.cuda.reset_peak_memory_stats()
    _log_vram("request-start")

    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    negative_prompt = (
        req.negative_prompt
        if req.negative_prompt is not None
        else DEFAULT_NEGATIVE_PROMPT
    )

    log.info(
        f"Generando: {req.width}x{req.height}, {req.num_frames} frames, "
        f"{req.num_inference_steps} steps, seed={seed}"
    )
    log.info(f"Prompt: {req.prompt[:120]}…" if len(req.prompt) > 120 else f"Prompt: {req.prompt}")

    _log_vram("pre-pipeline")

    output = _pipe(
        prompt=req.prompt,
        negative_prompt=negative_prompt,
        height=req.height,
        width=req.width,
        num_frames=req.num_frames,
        guidance_scale=req.guidance_scale,
        guidance_scale_2=req.guidance_scale_2,
        num_inference_steps=req.num_inference_steps,
        generator=generator,
    )

    frames = output.frames[0]   # lista de PIL Images o np.ndarray frames
    _log_vram("post-pipeline")

    # Exportar a MP4 y leer los bytes
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_to_video(frames, tmp_path, fps=req.fps)
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    _log_vram("post-encode")
    log.info(f"✅ Vídeo generado: {len(video_bytes) / 1024:.1f} KB")

    return Response(content=video_bytes, media_type="video/mp4")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

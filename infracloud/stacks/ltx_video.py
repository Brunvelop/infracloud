"""
Built-in stack: LTX-Video 2.3

Deploys a FastAPI server that serves the Lightricks LTX-Video model for
text-to-video generation.

Endpoints once running:
  GET  /health    -> {"status": "ready", "model": "ltx-video"} (200 when model loaded)
  POST /generate  -> returns an MP4 video (video/mp4)

Example request:
    curl http://host:PORT/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "a cat walking on the moon"}' \\
      -o video.mp4
"""

from infracloud.stack import Stack
from infracloud.stacks import register

# ---- onstart script ----------------------------------------------------------
#
# This bash script runs automatically on the Vast.ai instance at boot time.
# It installs Python deps, writes the FastAPI server to disk, and launches it.
#
# The heredoc uses 'PYEOF' (single-quoted) so bash does NOT expand $variables
# inside the Python source — the $ characters are left for Python to handle.
#
# IMPORTANT: we intentionally avoid triple-quotes inside this string so that
# Python does not mistake the heredoc content for the end of this string.

_ONSTART = r"""#!/bin/bash
set -e

echo "[infracloud] Installing dependencies..."
source /venv/main/bin/activate
pip install --quiet --upgrade \
    torch \
    torchvision \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    fastapi \
    "uvicorn[standard]" \
    imageio \
    imageio-ffmpeg

echo "[infracloud] Writing server..."
mkdir -p /workspace

cat > /workspace/serve.py << 'PYEOF'
# LTX-Video FastAPI server -- written by infracloud onstart script.

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import imageio
import numpy as np
import torch
import uvicorn
from diffusers import LTXPipeline
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# ---- State ------------------------------------------------------------------

_pipe: Optional[LTXPipeline] = None
_model_ready: bool = False

MODEL_ID = "Lightricks/LTX-Video"


# ---- Lifespan ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe, _model_ready
    print(f"[serve] Loading model {MODEL_ID}...")
    _pipe = LTXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    _pipe = _pipe.to("cuda")
    _pipe.enable_model_cpu_offload()
    _model_ready = True
    print("[serve] Model ready.")
    yield


# ---- App --------------------------------------------------------------------

app = FastAPI(title="LTX-Video Server", lifespan=lifespan)


# ---- Schemas ----------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_frames: int = 97        # must be 8k+1 for LTX-Video (e.g. 25, 49, 97)
    width: int = 768
    height: int = 512
    num_inference_steps: int = 40
    guidance_scale: float = 7.5
    seed: Optional[int] = None


# ---- Endpoints --------------------------------------------------------------

@app.get("/health")
def health():
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model": "ltx-video"},
        )
    return {"status": "ready", "model": "ltx-video"}


@app.post("/generate")
def generate(req: GenerateRequest):
    if not _model_ready:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})

    generator = None
    if req.seed is not None:
        generator = torch.Generator("cuda").manual_seed(req.seed)

    result = _pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_frames=req.num_frames,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    )

    # result.frames[0] is a list of PIL Images (one per frame)
    frames = result.frames[0]

    # Encode frames to MP4 using a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = imageio.get_writer(tmp_path, fps=24, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return Response(content=video_bytes, media_type="video/mp4")


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
PYEOF

echo "[infracloud] Starting server on port 5000..."
source /venv/main/bin/activate
python /workspace/serve.py
"""

# ---- Stack definition --------------------------------------------------------

ltx_video = Stack(
    name="ltx-video",
    # vastai/base-image variants are pre-cached on Vast.ai hosts -> faster startup
    image="vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04",
    gpu_vram_gb=24,    # LTX-Video needs ~18 GB for standard resolution inference
    disk_gb=60,        # ~10 GB model weights + Python deps + output buffer
    ports=[5000],
    onstart=_ONSTART,
    health_url="/health",
)

register(ltx_video)

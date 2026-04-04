# LTX-2.3 FastAPI server
#
# This file is deployed to /workspace/serve.py on the Vast.ai instance at boot
# time by the ltx_video stack's onstart script. It is a real Python module so
# that IDEs, linters, and type-checkers can work with it normally.
#
# It is NOT imported directly by infracloud — it is read as package data by
# ltx_video.py and embedded into the bash onstart heredoc.
#
# Runtime dependencies (installed by the onstart script):
#   - xformers, flashpack
#   - transformers, accelerate, einops, scipy, av, scikit-image
#   - huggingface_hub
#   - fastapi, uvicorn
#   - ltx-core + ltx-pipelines (from https://github.com/Lightricks/LTX-2.git)

import os
import sys

# ---- Must happen before any torch import ------------------------------------
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Reduces CUDA allocator fragmentation; PyTorch's own OOM suggestion.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Add LTX-2 packages to path ---------------------------------------------
# The onstart script clones the LTX-2 repo to /workspace/LTX-2 and installs
# ltx-core and ltx-pipelines in editable mode. We also add the src paths so
# they are importable even before the egg-link propagates.
LTX_REPO_DIR = "/workspace/LTX-2"
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines", "src"))
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-core", "src"))

# ---- Torch setup (after env vars, before other LTX imports) -----------------
import torch

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Force-patch xformers memory-efficient attention into the LTX attention module.
# This significantly reduces VRAM usage on large models.
#
# xformers only supports compute capability ≤ 9.x (up to Hopper/H100).
# Blackwell (RTX 50xx, compute cap 12.x) and other future architectures will
# cause a NotImplementedError at runtime even though the import succeeds.
# We check the GPU capability before committing to the patch.
from ltx_core.model.transformer import attention as _attn_mod

try:
    from xformers.ops import memory_efficient_attention as _mea

    _xf_supported = True
    if torch.cuda.is_available():
        _cc_major, _cc_minor = torch.cuda.get_device_capability(0)
        if _cc_major >= 10:
            # Blackwell and newer are not yet supported by xformers.
            _xf_supported = False
            print(
                f"[serve] xformers patch skipped — GPU compute capability "
                f"{_cc_major}.{_cc_minor} not supported by xformers "
                f"(requires ≤ 9.x). Using LTX default attention."
            )

    if _xf_supported:
        _attn_mod.memory_efficient_attention = _mea
        print("[serve] xformers attention patch applied.")
except Exception as exc:
    print(f"[serve] xformers patch skipped ({exc}); using default attention.")

# ---- Standard library / third-party -----------------------------------------
import logging
import random
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video

logging.basicConfig(level=logging.INFO)

# ---- Model constants --------------------------------------------------------

LTX_MODEL_REPO = "Lightricks/LTX-2.3"
# Gemma 3 12B QAT checkpoint used as the text encoder by LTX-2.3.
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_FPS = 24.0

# ---- Helpers ----------------------------------------------------------------

def _log_memory(tag: str) -> None:
    """Log current VRAM usage (allocated / free / total)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(
            f"[VRAM {tag}] allocated={allocated:.2f}GB  peak={peak:.2f}GB  "
            f"free={free / 1024**3:.2f}GB  total={total / 1024**3:.2f}GB"
        )


def _snap32(n: int) -> int:
    """Round n up to the nearest multiple of 32."""
    return ((n + 31) // 32) * 32


def _valid_frames(duration: float, fps: float) -> int:
    """Return the smallest 8k+1 frame count that covers *duration* seconds."""
    raw = int(duration * fps) + 1
    k = (raw - 1 + 7) // 8
    return k * 8 + 1


# ---- State ------------------------------------------------------------------

_pipeline: Optional[DistilledPipeline] = None
_model_ready: bool = False


# ---- Lifespan ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _model_ready

    print("[serve] Downloading LTX-2.3 checkpoints from Hugging Face...")
    checkpoint_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename="ltx-2.3-22b-distilled.safetensors",
    )
    spatial_upsampler_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    )
    print("[serve] Downloading Gemma text encoder (this may take a while)...")
    gemma_root = snapshot_download(repo_id=GEMMA_REPO)

    print(f"[serve] checkpoint   : {checkpoint_path}")
    print(f"[serve] upsampler    : {spatial_upsampler_path}")
    print(f"[serve] gemma        : {gemma_root}")

    print("[serve] Initializing DistilledPipeline (FP8)...")
    _pipeline = DistilledPipeline(
        distilled_checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )

    # Eagerly load ALL sub-models so VRAM is allocated upfront in one controlled
    # pass — avoids fragmentation during the first generation request.
    # Pattern mirrors the official HF Spaces demo for LTX-2.3.
    # Wrapped in try/except: older builds of ltx-pipelines may not expose
    # model_ledger yet; in that case we skip preloading and let models load
    # lazily on the first request.
    print("[serve] Preloading all sub-models into VRAM...")
    _log_memory("before preload")
    try:
        ledger = _pipeline.model_ledger
        _transformer          = ledger.transformer()
        _video_encoder        = ledger.video_encoder()
        _video_decoder        = ledger.video_decoder()
        _audio_decoder        = ledger.audio_decoder()
        _vocoder              = ledger.vocoder()
        _spatial_upsampler    = ledger.spatial_upsampler()
        _text_encoder         = ledger.text_encoder()
        _embeddings_processor = ledger.gemma_embeddings_processor()

        # Replace the factory functions with direct references so repeated calls
        # always return the already-loaded instances (no re-allocation).
        ledger.transformer              = lambda: _transformer
        ledger.video_encoder            = lambda: _video_encoder
        ledger.video_decoder            = lambda: _video_decoder
        ledger.audio_decoder            = lambda: _audio_decoder
        ledger.vocoder                  = lambda: _vocoder
        ledger.spatial_upsampler        = lambda: _spatial_upsampler
        ledger.text_encoder             = lambda: _text_encoder
        ledger.gemma_embeddings_processor = lambda: _embeddings_processor

        _log_memory("after preload")
        print("[serve] All models preloaded into VRAM.")
    except AttributeError as _e:
        print(
            f"[serve] WARNING: model_ledger not available ({_e}). "
            "Skipping eager preload — models will load lazily on first request."
        )
        _log_memory("after preload (skipped)")
    _model_ready = True
    print("[serve] Pipeline ready — server accepting requests.")
    yield


# ---- App --------------------------------------------------------------------

app = FastAPI(title="LTX-2.3 Server", lifespan=lifespan)


# ---- Schemas ----------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    duration: float = 3.0          # seconds; final frame count is 8k+1
    width: int = 1536               # must be divisible by 32 (auto-snapped)
    height: int = 1024              # must be divisible by 32 (auto-snapped)
    enhance_prompt: bool = True     # uses Gemma to expand the prompt
    seed: Optional[int] = None      # random if None


# ---- Endpoints --------------------------------------------------------------

@app.get("/health")
def health():
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model": "ltx-2.3"},
        )
    return {"status": "ready", "model": "ltx-2.3"}


@app.post("/generate")
@torch.inference_mode()
def generate(req: GenerateRequest):
    """Generate an MP4 video from a text prompt.

    ``@torch.inference_mode()`` disables gradient tracking more aggressively
    than ``torch.no_grad()``, freeing the saved-tensor storage used for
    autograd — an important memory saving for a 22B model.
    """
    if not _model_ready:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})

    torch.cuda.reset_peak_memory_stats()
    _log_memory("request start")

    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    width = _snap32(req.width)
    height = _snap32(req.height)
    num_frames = _valid_frames(req.duration, DEFAULT_FPS)

    print(
        f"[serve] Generating: {width}x{height} {num_frames} frames "
        f"({req.duration:.1f}s) seed={seed}"
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    _log_memory("before pipeline call")

    video, audio = _pipeline(
        prompt=req.prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=DEFAULT_FPS,
        images=[],              # text-to-video: no image conditioning
        tiling_config=tiling_config,
        enhance_prompt=req.enhance_prompt,
    )

    _log_memory("after pipeline call")

    output_path = tempfile.mktemp(suffix=".mp4")
    try:
        encode_video(
            video=video,
            fps=DEFAULT_FPS,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        with open(output_path, "rb") as f:
            video_bytes = f.read()
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

    _log_memory("after encode_video")
    return Response(content=video_bytes, media_type="video/mp4")


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

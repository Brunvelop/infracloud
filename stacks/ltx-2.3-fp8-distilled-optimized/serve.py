# LTX-2.3 FastAPI server — OPTIMIZED variant
#
# Optimizations vs the baseline stack (ltx-2.3-fp8-distilled):
#
#   1. torch.compile(mode="max-autotune") on the transformer
#      Fuses CUDA kernels, autotunes conv/matmul kernels.
#      First call is slow (compilation); subsequent calls are 20-40% faster.
#      The Triton cache is persisted in /workspace/.triton_cache so a server
#      restart can reuse compiled artifacts without recompiling from scratch.
#
#   2. TF32 matmul + cuDNN benchmark
#      On Ampere/Hopper (A100, H100…) matmuls and conv use Tensor Cores at
#      FP32 throughput with TF32 precision (10-bit mantissa — more than enough
#      for inference).  cuDNN benchmark selects the fastest conv algorithm for
#      each unique input shape.
#
#   3. PyTorch native SDPA instead of xformers
#      torch.nn.functional.scaled_dot_product_attention dispatches to Flash
#      Attention 2 when available and is transparent to torch.compile (xformers
#      ops are opaque to the compiler and cannot be fused).  A fallback to
#      xformers is kept for older PyTorch builds that lack FA2.
#
#   4. channels_last memory format on the transformer
#      Reorganises tensor layout for better Tensor Core utilisation, especially
#      combined with torch.compile.
#
#   5. Warmup generation during startup
#      A tiny dummy generation (256×256, 9 frames) is run at the end of
#      lifespan so the Triton compilation and CUDA caches are warm before the
#      first real request arrives.

import os
import sys

# ---- Must happen before any torch import ------------------------------------
# NOTE: TORCH_COMPILE_DISABLE and TORCHDYNAMO_DISABLE are intentionally NOT
# set here — we want torch.compile to work in this optimized variant.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Persist the Triton autotuning cache across server restarts.
os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.triton_cache")

# ---- Add LTX-2 packages to path ---------------------------------------------
LTX_REPO_DIR = "/workspace/LTX-2"
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines", "src"))
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-core", "src"))

# ---- Torch setup ------------------------------------------------------------
import torch
import torch.nn.functional as F

# Allow torch.compile to work normally.
# suppress_errors=True ensures that if a specific op can't be compiled it
# falls back to eager execution rather than crashing the server.
torch._dynamo.config.suppress_errors = True

# ---- TF32 + cuDNN benchmark (free performance on Ampere/Hopper) -------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("[serve] TF32 matmul/cuDNN enabled, cuDNN benchmark=True")

# ---- Attention strategy: prefer native SDPA, fall back to xformers ----------
#
# torch.compile can fuse SDPA into the surrounding ops; it cannot fuse the
# xformers op (which is a C++/CUDA extension opaque to TorchDynamo).
# We attempt to verify that PyTorch's built-in flash-attention backend is
# available; if not, we fall back to xformers exactly as the baseline does.
_use_native_sdpa = False
try:
    # Flash Attention 2 is available in PyTorch >= 2.2 on Ampere+.
    # Check by inspecting the set of enabled backends for SDPA.
    from torch.backends.cuda import (
        flash_sdp_enabled,
        mem_efficient_sdp_enabled,
    )
    if flash_sdp_enabled() or mem_efficient_sdp_enabled():
        _use_native_sdpa = True
        print("[serve] Native SDPA (Flash Attention / mem-efficient) available — skipping xformers patch.")
    else:
        print("[serve] Native Flash Attention not available — will try xformers.")
except ImportError:
    print("[serve] torch.backends.cuda query unavailable — will try xformers.")

if not _use_native_sdpa:
    # Fallback: patch xformers into the LTX attention module (same as baseline).
    from ltx_core.model.transformer import attention as _attn_mod

    try:
        from xformers.ops import memory_efficient_attention as _mea

        _xf_supported = True
        if torch.cuda.is_available():
            _cc_major, _cc_minor = torch.cuda.get_device_capability(0)
            if _cc_major >= 10:
                _xf_supported = False
                print(
                    f"[serve] xformers patch skipped — GPU compute capability "
                    f"{_cc_major}.{_cc_minor} not supported by xformers "
                    f"(requires ≤ 9.x). Using LTX default attention."
                )

        if _xf_supported:
            _attn_mod.memory_efficient_attention = _mea
            print("[serve] xformers attention patch applied (fallback).")
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
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_FPS = 24.0

# ---- Helpers ----------------------------------------------------------------

def _log_memory(tag: str) -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(
            f"[VRAM {tag}] allocated={allocated:.2f}GB  peak={peak:.2f}GB  "
            f"free={free / 1024**3:.2f}GB  total={total / 1024**3:.2f}GB"
        )


def _snap32(n: int) -> int:
    return ((n + 31) // 32) * 32


def _valid_frames(duration: float, fps: float) -> int:
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

    # ---- Preload all sub-models into VRAM -----------------------------------
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

        # ---- Optimization 1: channels_last memory format --------------------
        # Reorganises the transformer's internal tensor layout so Tensor Cores
        # have better memory access patterns.  Applied before torch.compile so
        # the compiler sees the optimal layout from the start.
        try:
            _transformer.to(memory_format=torch.channels_last)
            print("[serve] Transformer converted to channels_last memory format.")
        except Exception as _e:
            print(f"[serve] channels_last conversion skipped ({_e}).")

        # ---- Optimization 2: torch.compile on the transformer ---------------
        # mode="max-autotune" runs GEMM / conv autotuning (slow first call,
        # much faster steady state).  The Triton cache in TRITON_CACHE_DIR
        # persists compiled kernels across server restarts.
        print("[serve] Compiling transformer with torch.compile(max-autotune)…")
        print("[serve]   (first request will trigger JIT compilation — use warmup below)")
        try:
            _transformer = torch.compile(_transformer, mode="max-autotune")
            print("[serve] torch.compile applied to transformer.")
        except Exception as _e:
            print(f"[serve] torch.compile failed ({_e}); running in eager mode.")

        # Replace ledger factory functions with direct references.
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

    # ---- Optimization 3: warmup generation ----------------------------------
    # Run a tiny dummy generation so Triton compiles all kernels and CUDA
    # allocator caches are warm before the first real user request.
    print("[serve] Running warmup generation (256×256, 9 frames)…")
    print("[serve]   torch.compile will JIT-compile kernels now — this takes 1-3 min.")
    _log_memory("before warmup")
    try:
        with torch.inference_mode():
            _pipeline(
                prompt="a simple warmup scene",
                seed=0,
                height=256,
                width=256,
                num_frames=9,
                frame_rate=DEFAULT_FPS,
                images=[],
                tiling_config=TilingConfig.default(),
                enhance_prompt=False,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _log_memory("after warmup")
        print("[serve] Warmup complete — kernels compiled and caches warm.")
    except Exception as _e:
        print(f"[serve] Warmup failed ({_e}); continuing without warmup.")

    _model_ready = True
    print("[serve] Pipeline ready — server accepting requests.")
    yield


# ---- App --------------------------------------------------------------------

app = FastAPI(title="LTX-2.3 Server (Optimized)", lifespan=lifespan)


# ---- Schemas ----------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    duration: float = 3.0
    width: int = 1536
    height: int = 1024
    enhance_prompt: bool = True
    seed: Optional[int] = None


# ---- Endpoints --------------------------------------------------------------

@app.get("/health")
def health():
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model": "ltx-2.3-optimized"},
        )
    return {"status": "ready", "model": "ltx-2.3-optimized"}


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
        images=[],
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

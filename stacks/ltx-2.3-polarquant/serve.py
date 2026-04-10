# LTX-2.3 FastAPI server — PolarQuant Q5 variant
#
# Flujo de startup:
#   1. Descarga PolarQuant Q5 desde HuggingFace (~15 GB, -68% vs FP16)
#   2. Dequantiza los códigos PQ5 a BF16 en GPU (~minutos, one-time)
#   3. Guarda el safetensors dequantizado en /workspace/
#   4. Carga DistilledPipeline con FP8 cast para inferencia eficiente
#
# Resultado: calidad similar a FP16 (cos_sim 0.9986) + inferencia en FP8.
#
# PolarQuant algorithm (Vicentino, 2026 — arXiv:2603.29078):
#   bhat = b / ||b||          # normalizar bloque a esfera unitaria
#   z    = sqrt(d) * H @ bhat # rotar con Hadamard → N(0,1)
#   q    = argmin_k |z - c_k| # cuantizar al centroide Lloyd-Max más cercano
#
# Dequantización (inversa):
#   z_hat = centroids[codes]  # lookup centroide
#   bhat  = H @ (z_hat / sqrt(d))  # H^-1 = H (autoinversa)
#   b     = norm * bhat       # restaurar norma del bloque

import os
import sys

# ---- Must happen before any torch import ------------------------------------
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Add LTX-2 packages to path ---------------------------------------------
LTX_REPO_DIR = "/workspace/LTX-2"
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines", "src"))
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-core", "src"))

# ---- Torch setup (after env vars, before other LTX imports) -----------------
import torch

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Patch xformers attention si está disponible y soportado.
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
                f"{_cc_major}.{_cc_minor} not supported (requires ≤ 9.x)."
            )

    if _xf_supported:
        _attn_mod.memory_efficient_attention = _mea
        print("[serve] xformers attention patch applied.")
except Exception as exc:
    print(f"[serve] xformers patch skipped ({exc}); using default attention.")

# ---- Standard library / third-party -----------------------------------------
import glob
import json
import logging
import math
import random
import tempfile
import time
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

PQ5_REPO      = "caiovicentino1/LTX-2.3-22B-PolarQuant-Q5"
LTX_REPO      = "Lightricks/LTX-2.3"
GEMMA_REPO    = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_FPS   = 24.0
BLOCK_SIZE    = 128          # PolarQuant block size (Hadamard dim)
DEQUANT_PATH  = "/workspace/ltx-pq5-dequant.safetensors"

# ---- PolarQuant dequantization ----------------------------------------------

_CENTROID_CACHE: dict[int, torch.Tensor] = {}


def _get_centroids(bits: int) -> torch.Tensor:
    """Compute (or return cached) Lloyd-Max centroids for N(0,1) at *bits* bits."""
    if bits in _CENTROID_CACHE:
        return _CENTROID_CACHE[bits]

    from scipy.stats import norm as sp_norm

    n = 1 << bits
    bd = torch.linspace(-4.0, 4.0, n + 1)
    ct = torch.zeros(n)
    for _ in range(100):
        for i in range(n):
            lo, hi = bd[i].item(), bd[i + 1].item()
            plo, phi = sp_norm.cdf(lo), sp_norm.cdf(hi)
            if phi - plo > 1e-12:
                ct[i] = (sp_norm.pdf(lo) - sp_norm.pdf(hi)) / (phi - plo)
            else:
                ct[i] = (lo + hi) / 2
        for i in range(1, n):
            bd[i] = (ct[i - 1] + ct[i]) / 2

    _CENTROID_CACHE[bits] = ct
    return ct


def _build_hadamard(n: int) -> torch.Tensor:
    """Build the normalised Walsh-Hadamard matrix of order *n* (must be 2^k)."""
    if n == 1:
        return torch.tensor([[1.0]])
    h = _build_hadamard(n // 2)
    top = torch.cat([h, h], dim=1)
    bot = torch.cat([h, -h], dim=1)
    return torch.cat([top, bot], dim=0) / math.sqrt(2)


def _unpack_5bit(packed_flat: torch.Tensor, total_elements: int) -> torch.Tensor:
    """Unpack 5-bit codes stored as 8 values per 5 bytes (40 bits per group)."""
    p = packed_flat.long().reshape(-1, 5)
    b0, b1, b2, b3, b4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
    codes = torch.stack([
        (b0 >> 3) & 31,
        ((b0 & 7) << 2) | ((b1 >> 6) & 3),
        (b1 >> 1) & 31,
        ((b1 & 1) << 4) | ((b2 >> 4) & 15),
        ((b2 & 15) << 1) | ((b3 >> 7) & 1),
        (b3 >> 2) & 31,
        ((b3 & 3) << 3) | ((b4 >> 5) & 7),
        b4 & 31,
    ], dim=-1).reshape(-1)
    return codes[:total_elements].to(torch.uint8)


def _dequantize_polarquant(pq5_dir: str, device: str) -> dict:
    """Dequantize PolarQuant Q5 codes to a BF16 state dict.

    Returns the full state dict ready for ``save_file`` / model loading.
    The dict contains both the dequantized transformer weights and the
    BF16-kept VAE / skip connection tensors.

    Args:
        pq5_dir: Local directory containing the downloaded PQ5 repo.
        device:  CUDA device to use for Hadamard matrix operations.

    Returns:
        dict mapping tensor names (e.g. ``"transformer.blocks.0.ff.weight"``)
        to BF16 tensors on CPU.
    """
    from safetensors.torch import load_file

    print("[dequant] Building Lloyd-Max centroids (5 bits, 100 iterations)...")
    ct5 = _get_centroids(5).to(device)

    print("[dequant] Building Hadamard matrix (128×128)...")
    H = _build_hadamard(BLOCK_SIZE).to(device)

    # ── Load codes (all chunks) ───────────────────────────────────────────────
    # Support both naming conventions:
    #   - New:  polar_state_chunk{N}.safetensors  (repo root)
    #   - Old:  polarquant/codes/chunk_{N}_codes.safetensors
    code_patterns = [
        os.path.join(pq5_dir, "polar_state_chunk*.safetensors"),
        os.path.join(pq5_dir, "polarquant", "codes", "chunk_*_codes.safetensors"),
    ]
    code_files: list[str] = []
    for pat in code_patterns:
        found = sorted(glob.glob(pat))
        if found:
            code_files = found
            break

    if not code_files:
        raise RuntimeError(
            f"No se encontraron archivos de códigos PolarQuant en {pq5_dir}. "
            "Verifica que el snapshot_download completó correctamente."
        )
    print(f"[dequant] Cargando {len(code_files)} chunk(s) de códigos...")

    codes_sd: dict[str, torch.Tensor] = {}
    for f in code_files:
        print(f"  {os.path.basename(f)}...")
        codes_sd.update(load_file(f, device="cpu"))

    # ── Load BF16 kept parts (VAE, skip connections, upscalers if included) ──
    bf16_patterns = [
        os.path.join(pq5_dir, "bf16_kept.safetensors"),
        os.path.join(pq5_dir, "polarquant", "bf16", "ltx23_bf16.safetensors"),
    ]
    bf16_path = next((p for p in bf16_patterns if os.path.exists(p)), None)
    if bf16_path is None:
        raise RuntimeError(
            f"No se encontró bf16_kept.safetensors en {pq5_dir}."
        )
    print(f"[dequant] Cargando BF16 kept: {os.path.basename(bf16_path)}...")
    full_sd = dict(load_file(bf16_path, device="cpu"))
    print(f"  {len(full_sd)} tensores BF16 cargados.")

    # ── Dequantize each coded tensor ──────────────────────────────────────────
    coded_keys = sorted(set(k[:-8] for k in codes_sd if k.endswith("__packed")))
    print(f"[dequant] Dequantizando {len(coded_keys)} tensores PQ5 → BF16...")

    t0 = time.monotonic()
    for idx, ck in enumerate(coded_keys):
        meta  = codes_sd[f"{ck}__meta"]
        of    = int(meta[0])   # output features (first dim)
        nb    = int(meta[1])   # number of blocks per row
        bs    = int(meta[2])   # block size (== BLOCK_SIZE)
        total = int(meta[3])   # total number of code elements

        # Unpack and move to GPU
        codes = _unpack_5bit(codes_sd[f"{ck}__packed"], total)
        codes = codes.reshape(of, nb, bs).to(device).long()
        norms = codes_sd[f"{ck}__norms"].to(device).float().unsqueeze(2)

        # Centroid lookup + scale to unit-variance Gaussian
        vals = ct5[codes] / math.sqrt(BLOCK_SIZE)   # shape: (of, nb, bs)

        # Apply inverse Hadamard rotation in VRAM-friendly chunks of 256 rows
        for i in range(0, of, 256):
            e = min(i + 256, of)
            vals[i:e] = (vals[i:e].reshape(-1, BLOCK_SIZE) @ H).reshape(e - i, nb, BLOCK_SIZE)

        # Restore block norms and flatten to original weight shape
        tensor_name = ck.replace("__", ".")
        full_sd[tensor_name] = (vals * norms).reshape(of, nb * bs).to(torch.bfloat16).cpu()

        # Free GPU tensors immediately to keep VRAM usage low
        del codes, norms, vals
        torch.cuda.empty_cache()

        if (idx + 1) % 100 == 0 or (idx + 1) == len(coded_keys):
            elapsed = time.monotonic() - t0
            print(f"  {idx + 1}/{len(coded_keys)} tensores ({elapsed:.0f}s)")

    elapsed_total = time.monotonic() - t0
    print(f"[dequant] Dequantización completada en {elapsed_total:.1f}s.")
    return full_sd


def _find_spatial_upsampler(pq5_dir: str) -> str | None:
    """Find the spatial upsampler safetensors in *pq5_dir* (several layouts)."""
    candidates = [
        os.path.join(pq5_dir, "upscalers", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        os.path.join(pq5_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        os.path.join(pq5_dir, "assets", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


# ---- Helpers ----------------------------------------------------------------

def _log_memory(tag: str) -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak      = torch.cuda.max_memory_allocated() / 1024**3
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pq5_dir = "/workspace/LTX-PQ5"

    # ── 1. Descarga PolarQuant Q5 (15 GB) ────────────────────────────────────
    print("[serve] Descargando PolarQuant Q5 desde HuggingFace (~15 GB)...")
    print(f"  Repo: {PQ5_REPO}")
    print(f"  Destino: {pq5_dir}")
    snapshot_download(repo_id=PQ5_REPO, local_dir=pq5_dir)
    print("[serve] Descarga completada.")
    _log_memory("after pq5 download")

    # ── 2. Dequantizar PQ5 → BF16 ────────────────────────────────────────────
    if os.path.exists(DEQUANT_PATH):
        print(f"[serve] Safetensors dequantizado encontrado en {DEQUANT_PATH}, saltando dequant.")
    else:
        print("[serve] Dequantizando PQ5 → BF16 (one-time, ~5-10 min)...")
        _log_memory("before dequant")
        full_sd = _dequantize_polarquant(pq5_dir, device)
        _log_memory("after dequant (pre-save)")

        print(f"[serve] Guardando {len(full_sd)} tensores en {DEQUANT_PATH}...")
        from safetensors.torch import save_file
        save_file(full_sd, DEQUANT_PATH)
        del full_sd
        torch.cuda.empty_cache()
        sz_gb = os.path.getsize(DEQUANT_PATH) / 1e9
        print(f"[serve] Guardado: {DEQUANT_PATH} ({sz_gb:.1f} GB)")
        _log_memory("after dequant save")

    # ── 3. Spatial upsampler — PQ5 repo o fallback a LTX-2.3 oficial ─────────
    spatial_upsampler_path = _find_spatial_upsampler(pq5_dir)
    if spatial_upsampler_path:
        print(f"[serve] Spatial upsampler encontrado en: {spatial_upsampler_path}")
    else:
        print("[serve] Spatial upsampler no encontrado en PQ5 repo, descargando de LTX-2.3...")
        spatial_upsampler_path = hf_hub_download(
            repo_id=LTX_REPO,
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        )
        print(f"[serve] Descargado: {spatial_upsampler_path}")

    # ── 4. Gemma text encoder ─────────────────────────────────────────────────
    print("[serve] Descargando Gemma text encoder (esto puede tardar)...")
    gemma_root = snapshot_download(repo_id=GEMMA_REPO)
    print(f"[serve] Gemma: {gemma_root}")

    # ── 5. Inicializar DistilledPipeline con FP8 cast ─────────────────────────
    # PolarQuant dequantizó los pesos a BF16 de alta calidad.
    # Aplicamos FP8 cast en runtime para mantener la velocidad de inferencia
    # del stack original — mejor calidad PQ5 + misma velocidad FP8.
    print("[serve] Inicializando DistilledPipeline (PQ5-dequant + FP8 cast)...")
    _log_memory("before pipeline init")
    _pipeline = DistilledPipeline(
        distilled_checkpoint_path=DEQUANT_PATH,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )

    # ── 6. Precargar todos los submodelos en VRAM ─────────────────────────────
    print("[serve] Precargando submodelos en VRAM...")
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

        ledger.transformer              = lambda: _transformer
        ledger.video_encoder            = lambda: _video_encoder
        ledger.video_decoder            = lambda: _video_decoder
        ledger.audio_decoder            = lambda: _audio_decoder
        ledger.vocoder                  = lambda: _vocoder
        ledger.spatial_upsampler        = lambda: _spatial_upsampler
        ledger.text_encoder             = lambda: _text_encoder
        ledger.gemma_embeddings_processor = lambda: _embeddings_processor

        _log_memory("after preload")
        print("[serve] Todos los modelos precargados en VRAM.")
    except AttributeError as _e:
        print(
            f"[serve] WARNING: model_ledger no disponible ({_e}). "
            "Los modelos se cargarán lazy en el primer request."
        )
        _log_memory("after preload (skipped)")

    _model_ready = True
    print("[serve] Pipeline listo — servidor aceptando requests.")
    yield


# ---- App --------------------------------------------------------------------

app = FastAPI(title="LTX-2.3 Server (PolarQuant Q5)", lifespan=lifespan)


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
            content={"status": "loading", "model": "ltx-2.3-polarquant"},
        )
    return {"status": "ready", "model": "ltx-2.3-polarquant"}


@app.post("/generate")
@torch.inference_mode()
def generate(req: GenerateRequest):
    """Genera un vídeo MP4 desde un prompt de texto.

    ``@torch.inference_mode()`` desactiva el tracking de gradientes de forma
    más agresiva que ``torch.no_grad()``, liberando el almacenamiento de tensores
    guardados para autograd — ahorro de memoria importante en un modelo de 22B.
    """
    if not _model_ready:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})

    torch.cuda.reset_peak_memory_stats()
    _log_memory("request start")

    seed       = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    width      = _snap32(req.width)
    height     = _snap32(req.height)
    num_frames = _valid_frames(req.duration, DEFAULT_FPS)

    print(
        f"[serve] Generando: {width}x{height} {num_frames} frames "
        f"({req.duration:.1f}s) seed={seed}"
    )

    tiling_config      = TilingConfig.default()
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

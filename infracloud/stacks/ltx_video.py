"""
Built-in stack: LTX-2.3

Deploys a FastAPI server that serves Lightricks LTX-2.3 for text-to-video
generation with synchronized audio. Uses the native LTX-2 codebase (not
diffusers) with the distilled 22B model, FP8 quantization, spatial upsampler
(x2) and Gemma 3 12B as text encoder.

Endpoints once running:
  GET  /health    -> {"status": "ready", "model": "ltx-2.3"} (200 when ready)
  POST /generate  -> returns an MP4 video with audio (video/mp4)

Example request:
    curl http://host:PORT/generate \\
      -H "Content-Type: application/json" \\
      -d '{
        "prompt": "a cat walking on the moon",
        "duration": 3.0,
        "width": 1536,
        "height": 1024,
        "enhance_prompt": true
      }' \\
      -o video.mp4

Hardware requirements:
  - GPU VRAM: ≥ 48 GB  (22B model at FP8 + Gemma 12B + spatial upsampler)
  - Disk:     ≥ 200 GB (model ~50 GB + Gemma ~25 GB + deps + buffer)
  - CUDA:     ≥ 12.7
"""

import os

from dotenv import load_dotenv
from importlib.resources import files

from infracloud.stack import Stack
from infracloud.stacks import register

# Load .env so HF_TOKEN is available when building the Stack env dict
load_dotenv()

# ---- Server script -----------------------------------------------------------
#
# ltx_video_serve.py is a real Python module that lives alongside this file.
# It is read here as package data and injected verbatim into the bash onstart
# heredoc so the remote instance writes it to /workspace/serve.py at boot.
#
# We use plain string concatenation (not an f-string) to avoid conflicts with
# the curly braces that appear in the Python source code.

_SERVE_PY: str = (
    files("infracloud.stacks")
    .joinpath("ltx_video_serve.py")
    .read_text(encoding="utf-8")
)

# ---- onstart script ----------------------------------------------------------
#
# Runs at boot on the Vast.ai instance. Steps:
#   1. Install xformers (needs --no-build-isolation against the existing torch)
#   2. Install Python deps used by the LTX-2 pipeline and the serve script
#   3. Clone the LTX-2 monorepo and install ltx-core + ltx-pipelines
#   4. Write serve.py and start the FastAPI server
#
# The heredoc sentinel 'PYEOF' (single-quoted) prevents bash from expanding
# $variables inside the Python source — the $ chars are left for Python.
#
# TORCH_COMPILE_DISABLE / TORCHDYNAMO_DISABLE are passed as Docker env vars
# via Stack.env so they are set before the process starts; we also export them
# here in case the shell spawns child processes.

_ONSTART = (
    r"""#!/bin/bash
set -e

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
# Reduces CUDA allocator fragmentation (PyTorch's own OOM suggestion).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[infracloud] Activating venv..."
source /venv/main/bin/activate

# xformers must be installed against the torch already in the venv.
# --no-build-isolation ensures it links against the correct CUDA/torch headers.
echo "[infracloud] Installing xformers..."
pip install --quiet xformers --no-build-isolation

echo "[infracloud] Installing Python dependencies..."
pip install --quiet \
    "transformers==4.57.6" \
    accelerate \
    einops \
    scipy \
    av \
    "scikit-image>=0.25.2" \
    flashpack \
    fastapi \
    "uvicorn[standard]" \
    "huggingface_hub[hf_transfer]"

# Enable faster HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[infracloud] Cloning/updating LTX-2 repository..."
if [ -d /workspace/LTX-2 ]; then
    # Repo already exists — pull latest commits so model_ledger and other
    # recent API additions are always available.
    git -C /workspace/LTX-2 pull --ff-only || true
else
    git clone --depth 1 https://github.com/Lightricks/LTX-2.git /workspace/LTX-2
fi

echo "[infracloud] Installing ltx-core and ltx-pipelines..."
# --force-reinstall --no-deps mirrors the approach used by the official Gradio
# demo to avoid dependency conflicts with the pre-installed torch environment.
pip install --quiet --force-reinstall --no-deps \
    -e /workspace/LTX-2/packages/ltx-core \
    -e /workspace/LTX-2/packages/ltx-pipelines

echo "[infracloud] Writing server..."
mkdir -p /workspace

cat > /workspace/serve.py << 'PYEOF'
"""
    + _SERVE_PY
    + r"""PYEOF

echo "[infracloud] Starting server on port 5000..."
source /venv/main/bin/activate
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
python /workspace/serve.py
"""
)

# ---- Stack definition --------------------------------------------------------

ltx_video = Stack(
    name="ltx-video",
    # Official Vast.ai PyTorch image with @vastai-automatic-tag.
    # The template hash tells Vast.ai to auto-select the right CUDA tag for the
    # target host (cuda-12.9.1-auto, cuda-12.4.1-auto, …).
    image="vastai/pytorch",
    template_hash="b84ca276fa572e949cd7ff43ae5fe855",  # "PyTorch (Vast)" template
    gpu_vram_gb=48,    # 22B FP8 + Gemma 12B + spatial upsampler
    disk_gb=200,       # ~50 GB LTX model + ~25 GB Gemma + deps + buffer
    ports=[5000],
    onstart=_ONSTART,
    health_url="/health",
    min_cuda_ver=12.7,  # LTX-2 requires CUDA ≥ 12.7
    env={
        # Disable torch.compile/dynamo globally — LTX-2 requires this
        "TORCH_COMPILE_DISABLE": "1",
        "TORCHDYNAMO_DISABLE": "1",
        # Reduces CUDA allocator fragmentation; avoids OOM on 48 GB GPUs
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Faster Hugging Face downloads via hf_transfer
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # HuggingFace token — required to download gated models (e.g. Gemma).
        # Read from the local environment / .env file at import time.
        **({ "HF_TOKEN": os.environ["HF_TOKEN"] } if os.environ.get("HF_TOKEN") else {}),
    },
)

register(ltx_video)

"""
Built-in stack: ComfyUI

Uses Vast.ai's official pre-built ComfyUI image (vastai/comfy), which includes:
  - ComfyUI pre-installed at /opt/workspace-internal/ComfyUI
  - CUDA + PyTorch ready (via /venv/main virtualenv)
  - ComfyUI Manager for downloading models from the UI

Access once running:
  http://<URL>:8188   -> ComfyUI web interface (open in browser)

Models are NOT downloaded automatically — use the ComfyUI Manager in the UI
or add model download commands to a custom onstart script.

Default models directory inside the container:
  /opt/workspace-internal/ComfyUI/models/

Example custom stack with model pre-download:

    from infracloud.stack import Stack
    from infracloud.stacks.comfyui import comfyui

    from dataclasses import replace

    my_comfyui = replace(comfyui, onstart=comfyui.onstart + '''
    # Download SDXL base model
    wget -q -O /opt/workspace-internal/ComfyUI/models/checkpoints/sdxl_base.safetensors \\
      https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/...
    ''')
"""

from infracloud.stack import Stack
from infracloud.stacks import register

# ---- onstart script ----------------------------------------------------------
#
# The vastai/comfy image has ComfyUI pre-installed at
# /opt/workspace-internal/ComfyUI, with a Python venv at /venv/main.
#
# The image sets COMFYUI_ARGS to use port 18188 by default; we override that
# to use 8188 so it matches the mapped container port.

_ONSTART = r"""#!/bin/bash
set -e

echo "[infracloud] Starting ComfyUI..."

# Activate the pre-built venv that ships with the vastai/comfy image
source /venv/main/bin/activate

# Kill any existing ComfyUI process (in case the image auto-started one)
pkill -f "main.py" 2>/dev/null || true
sleep 2

# ComfyUI lives at /opt/workspace-internal/ComfyUI in the vastai/comfy image
COMFYUI_DIR="/opt/workspace-internal/ComfyUI"

nohup python "${COMFYUI_DIR}/main.py" \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    > /var/log/comfyui.log 2>&1 </dev/null &

echo "[infracloud] ComfyUI started on port 8188. Logs at /var/log/comfyui.log"
echo "[infracloud] Models directory: ${COMFYUI_DIR}/models/"
"""

# ---- Stack definition --------------------------------------------------------

comfyui = Stack(
    name="comfyui",
    # Official Vast.ai ComfyUI image with @vastai-automatic-tag.
    # The template (hash below) tells Vast.ai to auto-select the right
    # CUDA tag for the target host.
    image="vastai/comfy",
    template_hash="7a57409b680b49a49a176abf22879b74",  # "ComfyUI" template (id 325864)
    gpu_vram_gb=32,     # 32 GB minimum — comfortable for SDXL/Flux/LTX
    disk_gb=100,        # Models can be large: SDXL ~7GB, Flux ~24GB, etc.
    ports=[8188],
    onstart=_ONSTART,
    health_url="/queue",    # ComfyUI returns {"queue_running": [], "queue_pending": []}
    min_cuda_ver=12.9,      # minimum workload requirement — used for host filtering
)

register(comfyui)

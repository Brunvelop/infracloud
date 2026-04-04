#!/bin/bash
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

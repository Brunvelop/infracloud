#!/bin/bash
# onstart para comfyui-ltx23-10eros
#
# Estrategia: arrancar ComfyUI INMEDIATAMENTE para que el health check pase
# rápido y el usuario tenga la URL en ~2 min. El provisioning (custom nodes +
# ~70GB de modelos) corre en background y reinicia ComfyUI cuando los nodos
# están instalados.
#
# Progreso:  tail -f /var/log/provision.log
# ComfyUI:   tail -f /var/log/comfyui.log

set -e

COMFY="/opt/workspace-internal/ComfyUI"
COMFY_LOG="/var/log/comfyui.log"
PROVISION_LOG="/var/log/provision.log"

source /venv/main/bin/activate

# ─── 1. Arrancar ComfyUI ya (URL disponible cuanto antes) ─────────────────────

echo "[infracloud] Starting ComfyUI (initial)..."
pkill -f "ComfyUI/main.py" 2>/dev/null || true
sleep 2
nohup python "${COMFY}/main.py" \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    >> "${COMFY_LOG}" 2>&1 </dev/null &

# ─── 2. Escribir el script de provisioning y lanzarlo en background ───────────
# Heredoc con comillas ('PROVEOF') → nada se expande aquí; el script lee
# HF_TOKEN e INFRACLOUD_REPO_URL del entorno del contenedor en runtime.

cat > /root/provision.sh << 'PROVEOF'
#!/bin/bash
# Provisioning de comfyui-ltx23-10eros — corre en background tras el arranque.
set +e

COMFY="/opt/workspace-internal/ComfyUI"
COMFY_LOG="/var/log/comfyui.log"
STACK_DIR="/workspace/infracloud/stacks/comfyui-ltx23-10eros"

source /venv/main/bin/activate

start_comfyui() {
    pkill -f "ComfyUI/main.py" 2>/dev/null || true
    sleep 3
    nohup python "${COMFY}/main.py" \
        --listen 0.0.0.0 \
        --port 8188 \
        --enable-cors-header \
        >> "${COMFY_LOG}" 2>&1 </dev/null &
    echo "[provision] ComfyUI reiniciado en :8188"
}

# download <url> <dest_path> — idempotente, reintentos, token HF
download() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "[skip] $(basename "$dest") ya existe"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    echo "[download] $(basename "$dest")"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 8 -s 8 --max-tries=5 --retry-wait=10 --continue=true \
            --console-log-level=warn --summary-interval=30 \
            --header="Authorization: Bearer ${HF_TOKEN}" \
            -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url" \
            && return 0
        echo "[warn] aria2c falló — reintentando con wget"
        rm -f "$dest"
    fi
    wget -q -c --tries=5 \
        --header="Authorization: Bearer ${HF_TOKEN}" \
        -O "$dest" "$url" || { echo "[ERROR] fallo descargando $url"; rm -f "$dest"; return 1; }
}

echo "════════════════════════════════════════════════════"
echo "[provision] START $(date)"

# 2.0 Herramientas de descarga
apt-get install -y -q aria2 >/dev/null 2>&1 || true

# 2.1 Clonar el repo infracloud (contiene models.txt / nodes.txt / workflow.json)
if [ ! -d /workspace/infracloud ]; then
    echo "[provision] Clonando repo: ${INFRACLOUD_REPO_URL}"
    git clone --depth 1 "${INFRACLOUD_REPO_URL}" /workspace/infracloud \
        || { echo "[ERROR] No se pudo clonar el repo. Abortando provisioning."; exit 1; }
fi

# 2.2 Instalar el workflow como default del usuario
mkdir -p "${COMFY}/user/default/workflows"
cp "${STACK_DIR}/workflow.json" "${COMFY}/user/default/workflows/LTX23-10Eros-Director.json"
echo "[provision] Workflow instalado en user/default/workflows/"

# 2.3 Custom nodes — primero cm-cli (registry de ComfyUI-Manager), fallback git clone
CM_CLI=""
for c in "${COMFY}/custom_nodes/ComfyUI-Manager/cm-cli.py" "${COMFY}/custom_nodes/comfyui-manager/cm-cli.py"; do
    [ -f "$c" ] && CM_CLI="$c" && break
done

grep -vE '^\s*(#|$)' "${STACK_DIR}/nodes.txt" | while IFS='|' read -r cnr_id git_url; do
    cnr_id="$(echo "$cnr_id" | xargs)"
    git_url="$(echo "$git_url" | xargs)"
    [ -z "$cnr_id" ] && continue

    # ¿Ya instalado? (por cnr_id o por nombre de repo, case-insensitive)
    if find "${COMFY}/custom_nodes" -maxdepth 1 -iname "$cnr_id" | grep -q . \
       || { [ -n "$git_url" ] && find "${COMFY}/custom_nodes" -maxdepth 1 -iname "$(basename "$git_url")" | grep -q .; }; then
        echo "[nodes] $cnr_id ya instalado — skip"
        continue
    fi

    installed=0
    if [ -n "$CM_CLI" ]; then
        echo "[nodes] cm-cli install $cnr_id"
        python "$CM_CLI" install "$cnr_id" 2>&1 | tail -3
        find "${COMFY}/custom_nodes" -maxdepth 1 -iname "*${cnr_id}*" | grep -q . && installed=1
    fi

    if [ "$installed" = "0" ] && [ -n "$git_url" ]; then
        dir="${COMFY}/custom_nodes/$(basename "$git_url")"
        echo "[nodes] git clone $git_url"
        git clone --depth 1 "$git_url" "$dir" || { echo "[WARN] fallo clonando $git_url"; continue; }
        if [ -f "$dir/requirements.txt" ]; then
            pip install -q -r "$dir/requirements.txt" || echo "[WARN] requirements de $cnr_id fallaron"
        fi
        [ -f "$dir/install.py" ] && python "$dir/install.py" 2>&1 | tail -2
    fi
done
echo "[provision] Custom nodes instalados"

# 2.4 Reiniciar ComfyUI para cargar los nodos nuevos
start_comfyui

# 2.5 Descargar modelos (~70GB). Los selectores de la UI se pueblan al refrescar (tecla R).
grep -vE '^\s*(#|$)' "${STACK_DIR}/models.txt" | while IFS='|' read -r url folder name; do
    url="$(echo "$url" | xargs)"
    folder="$(echo "$folder" | xargs)"
    name="$(echo "$name" | xargs)"
    [ -z "$url" ] && continue
    download "$url" "${COMFY}/models/${folder}/${name}"
done
echo "[provision] Modelos descargados"

echo "[provision] DONE $(date)"
echo "════════════════════════════════════════════════════"
PROVEOF

chmod +x /root/provision.sh
nohup bash /root/provision.sh >> "${PROVISION_LOG}" 2>&1 </dev/null &

echo "[infracloud] ComfyUI up on :8188 — provisioning en background."
echo "[infracloud] Progreso: tail -f ${PROVISION_LOG}"

# LTX-2.3 con infracloud

Guía completa para lanzar un servidor de generación de vídeo + audio en la nube (Vast.ai) y descargar los vídeos en local.

El servidor expone una API HTTP mínima: el vídeo generado se devuelve directamente como bytes en la respuesta, sin almacenamiento en el cloud. Todo queda en tu máquina.

**Modelo:** [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — 22B parámetros, genera vídeo + audio sincronizado. Usa la versión distilled (8 pasos), FP8 quantization, spatial upsampler ×2 y Gemma 3 12B como text encoder.

---

## Requisitos previos

- Python 3.11+ e `infracloud` instalado (`uv pip install -e .`)
- Cuenta en [Vast.ai](https://vast.ai) con crédito cargado
- API key de Vast.ai → guárdala en `.env`:

```bash
cp .env.example .env
# edita .env y añade:
# VAST_API_KEY=tu_api_key
```

O expórtala directamente:

```bash
export VAST_API_KEY=tu_api_key
```

---

## 1. Lanzar el servidor

```bash
infracloud up ltx-video
```

Esto hace automáticamente:
1. Busca la GPU más barata con ≥48 GB VRAM y CUDA ≥12.7 en Vast.ai
2. Crea la instancia y espera a que arranque
3. Instala xformers, flashpack y el resto de dependencias Python
4. Clona el repositorio [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) e instala `ltx-core` y `ltx-pipelines`
5. Descarga los checkpoints desde Hugging Face:
   - Modelo distilled 22B (`ltx-2.3-22b-distilled.safetensors`, ~50 GB)
   - Spatial upsampler ×2 (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`)
   - Gemma 3 12B text encoder (`google/gemma-3-12b-it-qat-q4_0-unquantized`, ~25 GB)
6. Lanza un servidor FastAPI en el puerto 5000
7. Espera hasta que el servidor responda correctamente al health check

**Tiempo típico: 20–35 minutos** (la mayor parte es la descarga de los modelos ~75 GB).

Salida esperada:

```
🔍 Buscando GPU con ≥48GB VRAM y CUDA ≥12.7...
✓ Encontrada: RTX A6000 · $0.65/hr · 48GB VRAM
🚀 Creando instancia...
  ID de instancia: 34018141
⏳ Esperando a que la instancia arranque...
⏳ Esperando a que el servidor esté listo... (esto puede tardar varios minutos)
✓ Servidor listo!

  URL:  http://175.155.64.174:19528
  SSH:  ssh -p 18140 root@ssh3.vast.ai
  Cost: $0.65/hr
```

---

## 2. Generar vídeos en local

Una vez el servidor está listo, usa `curl` para generar un vídeo y descargarlo directamente a tu máquina. El endpoint devuelve el MP4 (con audio) en el cuerpo de la respuesta.

### Ejemplo básico

```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat walking on the moon, cinematic"}' \
  -o video.mp4
```

### Con todos los parámetros

```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat walking on the moon, cinematic, 4k",
    "duration": 5.0,
    "width": 1536,
    "height": 1024,
    "enhance_prompt": true,
    "seed": 42
  }' \
  -o video.mp4
```

### Referencia de parámetros

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `prompt` | `string` | **requerido** | Descripción del vídeo a generar |
| `duration` | `float` | `3.0` | Duración en segundos (el frame count se calcula automáticamente como `8k+1`) |
| `width` | `int` | `1536` | Ancho en píxeles (se redondea al múltiplo de 32 más cercano) |
| `height` | `int` | `1024` | Alto en píxeles (se redondea al múltiplo de 32 más cercano) |
| `enhance_prompt` | `bool` | `true` | Expande el prompt usando Gemma antes de la generación |
| `seed` | `int\|null` | `null` | Semilla para reproducibilidad. Aleatorio si `null` |

> **Nota:** LTX-2.3 usa el modelo distilled (8 pasos, CFG=1), por lo que no hay parámetros de `num_inference_steps` ni `guidance_scale`.

### Recetas rápidas

**Prueba rápida (~2–3 min, resolución estándar):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a dog running on the beach", "duration": 3.0, "width": 768, "height": 512}' \
  -o prueba.mp4
```

**Calidad estándar (~4–6 min, 1536×1024):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a sunset over the ocean, golden hour, cinematic", "duration": 3.0, "seed": 1234}' \
  -o sunset.mp4
```

**Máxima duración (~8–12 min, 1536×1024, 5s):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a futuristic city at night, neon lights, rain, blade runner style", "duration": 5.0}' \
  -o ciudad.mp4
```

### Generar varios vídeos en serie

```bash
#!/bin/bash
URL=$(infracloud url)

prompts=(
  "a horse galloping through a forest"
  "a rocket launching into space"
  "waves crashing on rocks, slow motion"
)

for i in "${!prompts[@]}"; do
  echo "Generando vídeo $((i+1))..."
  curl -s -X POST "$URL/generate" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${prompts[$i]}\", \"duration\": 3.0, \"seed\": $i}" \
    -o "video_${i}.mp4"
  echo "  Guardado: video_${i}.mp4"
done
```

---

## 3. Generar vídeos desde Python

```python
import httpx
import subprocess

# Obtener la URL del servidor activo
url = subprocess.check_output(["infracloud", "url"]).decode().strip()

with httpx.Client(timeout=600) as client:  # timeout generoso: modelo grande
    response = client.post(
        f"{url}/generate",
        json={
            "prompt": "a cat walking on the moon, cinematic, 4k",
            "duration": 3.0,
            "width": 1536,
            "height": 1024,
            "enhance_prompt": True,
            "seed": 42,
        },
    )
    response.raise_for_status()

with open("video.mp4", "wb") as f:
    f.write(response.content)

print(f"Vídeo guardado: {len(response.content) / 1024:.0f} KB")
```

---

## 4. Verificar que el servidor está listo

```bash
# Health check manual
curl $(infracloud url)/health

# Respuesta cuando está listo:
# {"status": "ready", "model": "ltx-2.3"}

# Respuesta cuando aún carga el modelo:
# {"status": "loading", "model": "ltx-2.3"}  (HTTP 503)
```

---

## 5. Ver estado y URL

```bash
infracloud status   # muestra GPU, coste, URL, uptime
infracloud url      # solo la URL (útil en scripts)
```

---

## 6. Apagar el servidor

```bash
infracloud down
```

> ⚠️ **Importante:** Las instancias de Vast.ai se facturan por hora mientras están activas. Ejecuta `infracloud down` cuando termines para dejar de pagar.

---

## Tiempos y costes de referencia

| GPU | VRAM | Precio típico | Tiempo arranque | Tiempo por vídeo (3s, 1536×1024) |
|---|---|---|---|---|
| RTX A6000 | 48 GB | ~$0.50–0.80/hr | ~25–35 min | ~4–6 min |
| RTX 6000 Ada | 48 GB | ~$0.80–1.20/hr | ~25–35 min | ~3–5 min |
| A100 SXM | 80 GB | ~$1.50–2.50/hr | ~25–35 min | ~2–3 min |
| H100 PCIe | 80 GB | ~$2.00–3.50/hr | ~25–35 min | ~1–2 min |

> Los precios varían según disponibilidad. Consulta [cloud.vast.ai](https://cloud.vast.ai).
> El tiempo de arranque incluye la descarga de ~75 GB de modelos (LTX-2.3 + Gemma).

---

## Resolución de problemas

### El servidor no arranca en 15 minutos
La descarga de modelos (~75 GB) puede tardar más de 15 minutos en instancias con ancho de banda limitado. Infracloud espera hasta 15 min en el health check, pero el servidor puede seguir arrancando. Opciones:

1. Verifica la instancia en [cloud.vast.ai](https://cloud.vast.ai) — busca una con buen ancho de banda
2. Destruye y relanza: `infracloud down && infracloud up ltx-video`
3. Conecta por SSH (`infracloud status` te da el comando) y monitoriza el log

### El vídeo generado está vacío o hay un error 503
- El servidor aún puede estar cargando el modelo. Espera y vuelve a intentarlo
- Verifica el estado: `curl $(infracloud url)/health`

### No hay GPU disponible con los requisitos
```bash
# Prueba con una GPU de 40 GB (rendimiento reducido, puede haber OOM)
infracloud up ltx-video --vram 40

# O con CUDA menos estricto
infracloud up ltx-video --min-cuda 12.4
```

### Error de VRAM durante la generación
LTX-2.3 con FP8 + spatial upsampler a 1536×1024 necesita ~45–48 GB.
Si hay OOM, reduce la resolución en la petición:

```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "...", "width": 1024, "height": 768}' \
  -o video.mp4
```

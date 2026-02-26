"""
FastAPI server for Stable Diffusion Image-to-Image WebUI.
Run: python app.py
Visit: http://localhost:7860
"""

import os

# ---------------------------------------------------------------------------
# Redirect model cache to D: drive (avoid filling C: drive)
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_CACHE_DIR, "hub")
os.environ["TORCH_HOME"] = os.path.join(_CACHE_DIR, "torch")

import io
import time
import glob
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

from pipeline import PipelineManager

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

app = FastAPI(title="SD Image-to-Image WebUI")
pipe_manager = PipelineManager()

# Serve static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round8(val: int) -> int:
    """Round to nearest multiple of 8."""
    return max(64, (val // 8) * 8)


def _make_filename() -> str:
    """Unique filename based on timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"output_{ts}.png"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/api/status")
async def status():
    """Return GPU / device info."""
    return pipe_manager.get_status()


@app.get("/api/history")
async def history():
    """Return the last 20 generated images (newest first)."""
    files = sorted(OUTPUTS_DIR.glob("output_*.png"), key=os.path.getmtime, reverse=True)
    items = []
    for f in files[:20]:
        items.append({
            "filename": f.name,
            "url": f"/api/outputs/{f.name}",
            "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return items


@app.get("/api/outputs/{filename}")
async def get_output(filename: str):
    """Serve a generated output image."""
    filepath = OUTPUTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(filepath), media_type="image/png")


@app.post("/api/generate")
async def generate(
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(25),
    cfg_scale: float = Form(7.5),
    seed: int = Form(-1),
    sampler: str = Form("dpm++"),
    strength: float = Form(0.55),
    face_strength: float = Form(0.6),
    quality_boost: bool = Form(True),
    hires_fix: bool = Form(False),
    hires_scale: float = Form(1.5),
    hires_strength: float = Form(0.45),
    image: UploadFile = File(None),
):
    """Generate an image using SD (text-to-image or img2img)."""

    # -- Validate prompt
    if not prompt.strip():
        raise HTTPException(400, "Prompt tidak boleh kosong!")

    # -- Validate & clamp params
    width = _round8(max(256, min(1024, width)))
    height = _round8(max(256, min(1024, height)))
    steps = max(1, min(100, steps))
    cfg_scale = max(1.0, min(30.0, cfg_scale))
    strength = max(0.1, min(1.0, strength))
    face_strength = max(0.0, min(1.0, face_strength))

    # -- Handle image upload for img2img
    reference_image = None
    if image is not None and image.filename != "":
        # Check file size
        contents = await image.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"Ukuran file melebihi batas {MAX_UPLOAD_BYTES // (1024*1024)} MB!")

        # Validate image format
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Format file tidak valid. Gunakan JPG atau PNG.")

        # Resize to target dimensions
        pil_image = pil_image.resize((width, height), Image.LANCZOS)
        reference_image = pil_image

    # -- Run pipeline
    try:
        result_image, used_seed = pipe_manager.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            seed=seed,
            sampler=sampler,
            reference_image=reference_image,
            strength=strength,
            face_strength=face_strength,
            quality_boost=quality_boost,
            hires_fix=hires_fix,
            hires_scale=hires_scale,
            hires_strength=hires_strength,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    # -- Save output
    filename = _make_filename()
    output_path = OUTPUTS_DIR / filename
    result_image.save(str(output_path), "PNG")

    # -- Return image as response
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)

    return JSONResponse({
        "success": True,
        "filename": filename,
        "url": f"/api/outputs/{filename}",
        "seed": used_seed,
        "width": width,
        "height": height,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, port=7860)

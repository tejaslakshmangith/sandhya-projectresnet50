"""
FastAPI backend for SmartMine ResNet-50 inference.

Start server:
    uvicorn api_resnet50:app --reload --port 8000

Endpoints:
    POST /predict   – upload an image file, returns class + confidence
    GET  /health    – liveness check
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Resolve the ai-model directory absolutely so imports work from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai-model"))

from inference_resnet50 import predict_image  # noqa: E402

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartMine AI API",
    description="ResNet-50 image classification for mine safety detection",
    version="1.0.0",
)

# ── CORS (allow Next.js dev server) ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness probe."""
    return {"status": "ok", "model": "resnet50_smartmine"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and receive a safety prediction.

    Returns:
        JSON: { "class": "safe"|"unsafe", "confidence": float }
    """
    # Validate content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Save upload to a temporary file
    filename = file.filename or "upload"
    suffix = os.path.splitext(filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_image(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    return result

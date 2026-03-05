"""
Inference module for SmartMineResNet50.

Usage:
    from inference_resnet50 import predict_image
    result = predict_image("path/to/image.jpg")
    # {"class": "unsafe", "confidence": 0.97}
"""

import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from models.resnet50_model import SmartMineResNet50

# ── Configuration ─────────────────────────────────────────────────────────────
CLASS_NAMES   = ["safe", "unsafe"]
# Resolve the weights path relative to this file so it works from any cwd
MODEL_WEIGHTS = Path(__file__).resolve().parent / "models" / "resnet50_smartmine.pth"

# ── Transform ─────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Lazy model loader ─────────────────────────────────────────────────────────
_model = None


def _load_model() -> SmartMineResNet50:
    """Load and cache the model on first call."""
    global _model
    if _model is None:
        _model = SmartMineResNet50(num_classes=len(CLASS_NAMES))
        _model.load_state_dict(
            torch.load(str(MODEL_WEIGHTS), map_location=torch.device("cpu"))
        )
        _model.eval()
    return _model


def predict_image(image_path: str) -> dict:
    """
    Run inference on a single image.

    Args:
        image_path: Path to the input image file.

    Returns:
        dict with keys:
            - "class"      : predicted class label (str)
            - "confidence" : prediction confidence (float, 0-1)
    """
    model = _load_model()

    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0)          # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)

    confidence, pred_idx = torch.max(probs, 1)

    return {
        "class":      CLASS_NAMES[pred_idx.item()],
        "confidence": round(float(confidence), 4),
    }

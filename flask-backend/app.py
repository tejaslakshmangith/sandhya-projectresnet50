"""
Flask backend for SmartMine Safety Detector.

Provides user management, image inference via ResNet-50, and prediction
history stored in SQLite through SQLAlchemy.

Start server:
    cd flask-backend && python app.py   (runs on port 5000)

Endpoints:
    POST   /api/users                   – Register a user
    GET    /api/users                   – List all users
    GET    /api/users/<id>              – Get a specific user
    GET    /api/users/<id>/predictions  – Get predictions for a user
    POST   /api/predict                 – Upload image → run inference → store result
    GET    /api/predictions             – Get all predictions (supports ?user_id=X)
    GET    /api/predictions/<id>        – Get a specific prediction
    DELETE /api/predictions/<id>        – Delete a prediction
    GET    /api/health                  – Health check
    GET    /api/stats                   – Summary statistics
"""

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# ---------------------------------------------------------------------------
# Resolve the ai-model directory so inference_resnet50 can be imported from
# any working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai-model"))

# ---------------------------------------------------------------------------
# App & extensions
# ---------------------------------------------------------------------------
app = Flask(__name__)

CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# SQLite database lives next to this file
DB_PATH = Path(__file__).resolve().parent / "smartmine.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))

    predictions = db.relationship("Prediction", backref="user", lazy=True,
                                  cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    image_filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))
    ip_address = db.Column(db.String(50), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_filename": self.image_filename,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "ip_address": self.ip_address,
        }


# ---------------------------------------------------------------------------
# Lazy import of inference module (avoids loading torch at startup if not used)
# ---------------------------------------------------------------------------
_predict_image = None


def get_predictor():
    global _predict_image
    if _predict_image is None:
        from inference_resnet50 import predict_image  # noqa: PLC0415
        _predict_image = predict_image
    return _predict_image


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_client_ip() -> str:
    """Return the best-effort client IP address."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or ""


# ---------------------------------------------------------------------------
# Routes — Health & Stats
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Liveness probe."""
    return jsonify({"status": "ok", "model": "resnet50_smartmine"}), 200


@app.get("/api/stats")
def stats():
    """Return summary statistics."""
    total = db.session.query(Prediction).count()
    safe_count = db.session.query(Prediction).filter_by(predicted_class="safe").count()
    unsafe_count = db.session.query(Prediction).filter_by(predicted_class="unsafe").count()

    # Per-user counts
    from sqlalchemy import func  # noqa: PLC0415
    per_user = (
        db.session.query(User.id, User.name, func.count(Prediction.id).label("count"))
        .outerjoin(Prediction, Prediction.user_id == User.id)
        .group_by(User.id)
        .all()
    )

    return jsonify({
        "total_predictions": total,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "per_user": [
            {"user_id": row.id, "user_name": row.name, "count": row.count}
            for row in per_user
        ],
    }), 200


# ---------------------------------------------------------------------------
# Routes — Users
# ---------------------------------------------------------------------------

@app.post("/api/users")
def create_user():
    """Register or return an existing user by email."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()

    if not name or not email:
        return jsonify({"error": "Both 'name' and 'email' are required."}), 400

    existing = db.session.query(User).filter_by(email=email).first()
    if existing:
        return jsonify(existing.to_dict()), 200

    user = User(name=name, email=email)
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201


@app.get("/api/users")
def list_users():
    """Return all registered users."""
    users = db.session.query(User).order_by(User.created_at.desc()).all()
    return jsonify([u.to_dict() for u in users]), 200


@app.get("/api/users/<int:user_id>")
def get_user(user_id: int):
    """Return a specific user."""
    user = db.session.get(User, user_id)
    if user is None:
        return jsonify({"error": "User not found."}), 404
    return jsonify(user.to_dict()), 200


@app.get("/api/users/<int:user_id>/predictions")
def get_user_predictions(user_id: int):
    """Return all predictions for a specific user."""
    user = db.session.get(User, user_id)
    if user is None:
        return jsonify({"error": "User not found."}), 404
    preds = (
        db.session.query(Prediction)
        .filter_by(user_id=user_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    return jsonify([p.to_dict() for p in preds]), 200


# ---------------------------------------------------------------------------
# Routes — Predictions
# ---------------------------------------------------------------------------

@app.post("/api/predict")
def predict():
    """
    Upload an image, run ResNet-50 inference, store and return the result.

    Accepts multipart/form-data with:
        file        – image file (required)
        user_id     – existing user id (optional int)
        user_name   – name to create/look up user (optional, used with user_email)
        user_email  – email to create/look up user (optional, used with user_name)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "No file selected."}), 400

    if not upload.content_type or not upload.content_type.startswith("image/"):
        return jsonify({"error": "File must be an image."}), 400

    # Resolve user_id --------------------------------------------------------
    user_id = None
    raw_uid = request.form.get("user_id")
    if raw_uid:
        try:
            user_id = int(raw_uid)
            if db.session.get(User, user_id) is None:
                return jsonify({"error": f"User {user_id} not found."}), 404
        except ValueError:
            return jsonify({"error": "'user_id' must be an integer."}), 400
    else:
        user_name = (request.form.get("user_name") or "").strip()
        user_email = (request.form.get("user_email") or "").strip()
        if user_name and user_email:
            user = db.session.query(User).filter_by(email=user_email).first()
            if not user:
                user = User(name=user_name, email=user_email)
                db.session.add(user)
                db.session.flush()
            user_id = user.id

    # Save upload to a temp file --------------------------------------------
    suffix = os.path.splitext(upload.filename)[-1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            upload.save(tmp)
            tmp_path = tmp.name

        predictor = get_predictor()
        ai_result = predictor(tmp_path)
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Store result -----------------------------------------------------------
    record = Prediction(
        user_id=user_id,
        image_filename=upload.filename,
        predicted_class=ai_result["class"],
        confidence=ai_result["confidence"],
        ip_address=_get_client_ip(),
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "class": ai_result["class"],
        "confidence": ai_result["confidence"],
        "prediction_id": record.id,
        "user_id": user_id,
    }), 200


@app.get("/api/predictions")
def list_predictions():
    """Return all predictions, optionally filtered by ?user_id=X."""
    query = db.session.query(Prediction)

    uid = request.args.get("user_id")
    if uid is not None:
        try:
            query = query.filter_by(user_id=int(uid))
        except ValueError:
            return jsonify({"error": "'user_id' must be an integer."}), 400

    preds = query.order_by(Prediction.created_at.desc()).all()
    return jsonify([p.to_dict() for p in preds]), 200


@app.get("/api/predictions/<int:pred_id>")
def get_prediction(pred_id: int):
    """Return a specific prediction."""
    pred = db.session.get(Prediction, pred_id)
    if pred is None:
        return jsonify({"error": "Prediction not found."}), 404
    return jsonify(pred.to_dict()), 200


@app.delete("/api/predictions/<int:pred_id>")
def delete_prediction(pred_id: int):
    """Delete a specific prediction record."""
    pred = db.session.get(Prediction, pred_id)
    if pred is None:
        return jsonify({"error": "Prediction not found."}), 404
    db.session.delete(pred)
    db.session.commit()
    return jsonify({"message": f"Prediction {pred_id} deleted."}), 200


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)

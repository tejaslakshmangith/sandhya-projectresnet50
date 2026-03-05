# SmartMine — ResNet-50 AI Safety Pipeline

A full-stack mine safety detection system combining a **Next.js frontend**, a **FastAPI backend**, a **Flask + SQLite backend**, and a **PyTorch ResNet-50 image classification model**.

---

## Architecture

```
User Upload Image
      ↓
Next.js Frontend  (lib/ai.ts)
      ↓  HTTP POST /predict
FastAPI Backend   (backend/api_resnet50.py)  ← inference only, port 8000
      ↓
ResNet-50 Model   (ai-model/models/resnet50_model.py)
      ↓
Prediction Result { class, confidence }

  — OR —

Next.js Frontend  (lib/ai.ts → saveUserAndPredict)
      ↓  POST /api/users  +  POST /api/predict
Flask Backend     (flask-backend/app.py)       ← inference + history, port 5000
      ↓
ResNet-50 Model + SQLite DB  (flask-backend/smartmine.db)
      ↓
Extended Result { class, confidence, prediction_id, user_id }
```

---

## Project Structure

```
.
├── ai-model/
│   ├── models/
│   │   └── resnet50_model.py      # SmartMineResNet50 class
│   ├── train_resnet50.py          # Training script
│   ├── inference_resnet50.py      # Inference helper
│   └── dataset/                   # Place training data here
│       ├── train/
│       │   ├── safe/
│       │   └── unsafe/
│       └── val/
│           ├── safe/
│           └── unsafe/
├── backend/
│   ├── api_resnet50.py            # FastAPI app (port 8000)
│   └── requirements.txt
├── flask-backend/
│   ├── app.py                     # Flask app with DB (port 5000)
│   └── requirements.txt
├── lib/
│   └── ai.ts                      # Next.js ↔ API bridge
└── ...                            # Next.js config, pages, etc.
```

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Prepare your dataset

Place labelled images in the `ai-model/dataset/` directory following the structure shown above.

### 3. Train the model

```bash
cd ai-model
python train_resnet50.py
```

The best checkpoint is saved to `ai-model/models/resnet50_smartmine.pth`.

### 4. Start the FastAPI server

```bash
cd backend
uvicorn api_resnet50:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 5. Start the Next.js frontend

```bash
npm install
npm run dev
```

Frontend: http://localhost:3000

---

## Flask Backend (with Database)

The Flask backend adds **user registration**, **prediction history**, and **statistics** on top of the same ResNet-50 inference engine.  It runs alongside the FastAPI server — both can be active at the same time.

### Install Flask dependencies

```bash
pip install -r flask-backend/requirements.txt
```

### Start the Flask server

```bash
cd flask-backend && python app.py
```

The server listens on **port 5000**.  A SQLite database (`flask-backend/smartmine.db`) is created automatically on first run.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Summary stats |
| `POST` | `/api/users` | Register / find a user `{ name, email }` |
| `GET` | `/api/users` | List all users |
| `GET` | `/api/users/<id>` | Get a specific user |
| `GET` | `/api/users/<id>/predictions` | Get all predictions for a user |
| `POST` | `/api/predict` | Upload image → inference → store → return result |
| `GET` | `/api/predictions` | Get all predictions (`?user_id=X` to filter) |
| `GET` | `/api/predictions/<id>` | Get a specific prediction |
| `DELETE` | `/api/predictions/<id>` | Delete a prediction record |

### Example curl commands

```bash
# Health check
curl http://localhost:5000/api/health

# Register a user
curl -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'

# Run inference (replace 1 with the returned user id)
curl -X POST http://localhost:5000/api/predict \
  -F "file=@/path/to/image.jpg" \
  -F "user_id=1"

# List all predictions
curl http://localhost:5000/api/predictions

# Filter by user
curl "http://localhost:5000/api/predictions?user_id=1"

# Stats
curl http://localhost:5000/api/stats

# Delete a prediction (replace 1 with the prediction id)
curl -X DELETE http://localhost:5000/api/predictions/1
```

### Database Schema

**`users`**

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER | Primary key |
| `name` | VARCHAR(120) | Display name |
| `email` | VARCHAR(200) | Unique identifier |
| `created_at` | DATETIME | UTC timestamp |

**`predictions`**

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER | Primary key |
| `user_id` | INTEGER | FK → users.id (nullable) |
| `image_filename` | VARCHAR(255) | Original filename |
| `predicted_class` | VARCHAR(20) | `"safe"` or `"unsafe"` |
| `confidence` | FLOAT | 0 – 1 |
| `created_at` | DATETIME | UTC timestamp |
| `ip_address` | VARCHAR(50) | Client IP (nullable) |

---

## Expected Accuracy

| Model     | Accuracy |
| --------- | -------- |
| ResNet-50 | 90–95%   |

---

## Environment Variables

| Variable                    | Default                  | Description                         |
| --------------------------- | ------------------------ | ----------------------------------- |
| `NEXT_PUBLIC_AI_API_URL`    | `http://localhost:8000`  | URL of the FastAPI backend          |
| `NEXT_PUBLIC_FLASK_API_URL` | `http://localhost:5000`  | URL of the Flask backend            |

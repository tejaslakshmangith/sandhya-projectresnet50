# SmartMine — ResNet-50 AI Safety Pipeline

A full-stack mine safety detection system combining a **Next.js frontend**, a **FastAPI backend**, and a **PyTorch ResNet-50 image classification model**.

---

## Architecture

```
User Upload Image
      ↓
Next.js Frontend  (lib/ai.ts)
      ↓  HTTP POST /predict
FastAPI Backend   (backend/api_resnet50.py)
      ↓
ResNet-50 Model   (ai-model/models/resnet50_model.py)
      ↓
Prediction Result { class, confidence }
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
│   ├── api_resnet50.py            # FastAPI app
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

## Expected Accuracy

| Model     | Accuracy |
| --------- | -------- |
| ResNet-50 | 90–95%   |

---

## Environment Variables

| Variable                  | Default                  | Description                    |
| ------------------------- | ------------------------ | ------------------------------ |
| `NEXT_PUBLIC_AI_API_URL`  | `http://localhost:8000`  | URL of the FastAPI backend     |

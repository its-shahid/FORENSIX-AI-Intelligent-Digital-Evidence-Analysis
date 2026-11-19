# Aegis Forensics — Deepfake Image Detection System

Interactive forensic web application built around your existing **EfficientNet-B0** PyTorch checkpoint (`models/best_model.pth`) and the `140k-real-fake-image` dataset conventions (`fake = 0`, `real = 1`).

---

## 1. Uploaded Project Inventory & Fixes

All original uploaded assets in `public/assets/aistudio/` are preserved intact:

| Original File | Role & Audit Findings | Integrated / Repaired Target |
| :--- | :--- | :--- |
| `app_422c9ef3.py` | Streamlit UI with EfficientNet-B0 (`224×224`, ImageNet norm) & Grad-CAM on `features[-1]` | Ported Grad-CAM into `backend/predict.py`; repaired Streamlit app in `scripts/streamlit_app.py` (added `class_to_idx` check, low-confidence warning cutoff, WEBP support). |
| `train_model_d6a819f4.py` | EfficientNet-B0 training (`Adam lr=1e-4`, `CrossEntropyLoss`, saves `model_state_dict`, `class_to_idx`, `image_size`) | Fixed `NoneType // 2` crash in `scripts/train_model.py`; defaults output to `models/retrained_best_model.pth` to protect `models/best_model.pth`. |
| `evaluate_model_92fbeaf8.py` | Balanced test evaluation & confusion matrix | Fixed module-level indentation bug (`NameError` on `full_test_data`) in `scripts/evaluate_model.py`; added `roc_auc` and JSON export (`results/evaluation_metrics.json`). |
| `check_predictions_a9ad7600.py` | Full test split accuracy check (`fake=0, real=1`) | Verified preprocessing parity & `weights_only=True` loading inside `backend/model_loader.py` and `scripts/verify_pipeline.py`. |
| `prepare_data_570eea58.py` | Dataset split loader | Added ImageNet normalization in `scripts/prepare_data.py` to match training & inference tensors. |
| `check_dataset_52d3b4d4.py` & `validate_dataset_369586fc.py` | Dataset split count & PIL `.verify()` checks | Integrity check pattern reused in `backend/preprocessing.py`. |
| `main_abe1b299.py` & `test_cbd7b9c3.py` | Local Python entry-point smoke scripts | Preserved in `public/assets/aistudio/`. |
| `settings_d0a49a74.json` | VS Code `git.ignoreLimitWarning` setting | Preserved in `public/assets/aistudio/`. |

---

## 2. Project Structure

```text
├── backend/
│   ├── __init__.py
│   ├── config.py            # Paths, class mapping (fake=0, real=1), normalization constants
│   ├── main.py              # FastAPI server (/api/health, /api/predict, /api/performance)
│   ├── model_loader.py      # Loads models/best_model.pth once at startup
│   ├── predict.py           # PyTorch inference, binary REAL/FAKE logic, Grad-CAM overlay
│   └── preprocessing.py     # PIL validation, RGB conversion, 224x224 ImageNet normalization
├── models/
│   ├── best_model.pth       # Place your existing trained PyTorch checkpoint here
│   └── model_config.json    # Canonical model architecture & normalization specification
├── results/                 # Output directory for evaluation_metrics.json & confusion_matrix.png
├── scripts/
│   ├── evaluate_model.py    # Generates real metrics + ROC-AUC + confusion matrix
│   ├── prepare_data.py      # Preserved dataset loader with matching normalization
│   ├── streamlit_app.py     # Repaired Streamlit + Grad-CAM app
│   ├── train_model.py       # Optional retraining script (writes separately from best_model.pth)
│   └── verify_pipeline.py   # Automated checkpoint, class mapping, & Grad-CAM test suite
├── src/                     # React + TypeScript + Tailwind CSS frontend (Aegis Forensics)
├── server.ts                # Express gateway on port 3000 (serves Vite/dist + proxies /api/*)
├── Dockerfile               # Cloud Run / Docker deployment for FastAPI + PyTorch CPU
├── render.yaml              # Render Web Service Blueprint
├── requirements.txt         # Python ML & FastAPI dependencies
└── package.json             # Node/Vite dependencies
```

---

## 3. Environment Variables

| Variable | Where Configured | Purpose |
| :--- | :--- | :--- |
| `VITE_PYTHON_BACKEND_URL` | Frontend Vite build / AI Studio environment | Deployed HTTPS FastAPI URL (e.g. `https://<your-service>.onrender.com`) called directly by the browser (`GET /health`, `POST /api/predict?threshold=0.65`). |
| `PYTHON_BACKEND_URL` | Node `server.ts` environment | Target URL when proxying same-origin `/api/*` requests through Express. |
| `MODEL_PATH` | FastAPI backend (`backend/config.py`) | Project-relative or absolute path to trained checkpoint (default: `models/best_model.pth`, or `models/improved_best_model.pth`). |
| `LOW_CONFIDENCE_THRESHOLD` | FastAPI backend (`backend/config.py`) | Low-confidence warning cutoff (default `0.65`). |
| `FAKE_DECISION_THRESHOLD` | FastAPI backend (`backend/config.py`) | Default `P(FAKE)` decision boundary (`0.50` = standard argmax). |
| `CORS_ORIGINS` | FastAPI backend (`backend/config.py`) | Comma-separated allowed browser origins (`*` by default for HTTPS previews). |

---

## 4. Permanent Render Cloud Deployment

1. Push `backend/`, `models/best_model.pth` (or `models/improved_best_model.pth`), `models/model_config.json`, `requirements.txt`, and `render.yaml` to GitHub.
2. Create a **Render Web Service** (or Blueprint from `render.yaml`) connected to your GitHub repo:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path**: `/health`
   - **Environment Variables**: `PYTHON_VERSION=3.10.14`, `MODEL_PATH=models/best_model.pth`, `LOW_CONFIDENCE_THRESHOLD=0.65`, `CORS_ORIGINS=*`
3. Verify `GET https://<your-service>.onrender.com/health` returns HTTP 200 with `"model_loaded": true`.
4. Set `VITE_PYTHON_BACKEND_URL=https://<your-service>.onrender.com` (and `PYTHON_BACKEND_URL` if using Express proxy) in the frontend environment, or paste the URL in **Backend Endpoint** in the UI.


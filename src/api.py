"""
api.py
------
FastAPI backend that serves breast cancer predictions.

Endpoints:
  GET  /                → serves the web UI
  POST /predict         → accepts 30 features, returns prediction
  GET  /feature-names   → returns the list of feature names
  GET  /health          → health check
"""

import os
import sys

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import get_data

# ── Train the model once on startup ──
X_train, X_test, y_train, y_test, FEATURE_NAMES, TARGET_NAMES = get_data()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
model.fit(X_train_s, y_train)

# Pre-compute test accuracy for the /health endpoint
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, model.predict(X_test_s))

print(f"Model trained! Test accuracy: {test_acc:.2%}")

# ── FastAPI app ──
app = FastAPI(title="Breast Cancer Predictor API")

# Serve static files (CSS, JS)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Request/Response schemas ──
class PredictionRequest(BaseModel):
    features: list[float]  # expects exactly 30 float values


class PredictionResponse(BaseModel):
    prediction: str            # "malignant" or "benign"
    confidence: float          # probability of the predicted class
    probability_benign: float
    probability_malignant: float


# ── Endpoints ──
@app.get("/")
def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/feature-names")
def get_feature_names():
    """Return the 30 feature names the model expects."""
    # Also return min/max/mean from training data for the UI sliders
    feature_info = []
    for i, name in enumerate(FEATURE_NAMES):
        feature_info.append({
            "name": name,
            "min": round(float(X_train[:, i].min()), 4),
            "max": round(float(X_train[:, i].max()), 4),
            "mean": round(float(X_train[:, i].mean()), 4),
        })
    return {"features": feature_info}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Accept 30 features and return a diagnosis prediction."""
    if len(req.features) != 30:
        return {"error": f"Expected 30 features, got {len(req.features)}"}

    # Scale input using the same scaler fitted on training data
    X_input = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # Get probabilities
    proba = model.predict_proba(X_scaled)[0]
    pred_class = int(np.argmax(proba))

    return PredictionResponse(
        prediction=TARGET_NAMES[pred_class],
        confidence=round(float(proba[pred_class]), 4),
        probability_benign=round(float(proba[1]), 4),
        probability_malignant=round(float(proba[0]), 4),
    )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "LogisticRegression",
        "test_accuracy": round(test_acc, 4),
        "num_features": len(FEATURE_NAMES),
    }

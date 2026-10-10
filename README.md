# Breast Cancer Diagnostic Classifier

An end-to-end machine learning project implementing, benchmarking, and serving diagnostic classifiers trained on the Wisconsin Breast Cancer Dataset (569 biopsy samples, 30 morphological cell features).

The project covers the complete ML lifecycle:
1. **Mathematical implementation from scratch** (vectorized NumPy: Sigmoid, Binary Cross-Entropy with L2 regularization, Gradient Descent).
2. **Scikit-learn production baseline & multi-model benchmarking** (Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting).
3. **Inference API** built with FastAPI and Pydantic validation schemas.
4. **Interactive diagnostic interface** built with Streamlit and real-time inference.

---

## Repository Structure

```
BreastCancerPredictor/
├── data/
│   └── load_data.py             # Dataset loading, train/test splitting (80/20)
├── src/
│   ├── logistic_from_scratch.py # NumPy vectorized Logistic Regression & Gradient Descent
│   ├── logistic_sklearn.py      # Scikit-learn Logistic Regression baseline
│   ├── compare_models.py        # 5-model benchmark suite & evaluation metrics
│   ├── utils.py                 # Feature scaling (Z-score), reporting, and plotting
│   └── api.py                   # FastAPI REST backend with Pydantic validation
├── app.py                       # Streamlit web interface with real-time prediction
└── README.md
```

---

## 1. Algorithm Implementation From Scratch

`src/logistic_from_scratch.py` implements binary logistic regression using vectorized NumPy operations:

- **Hypothesis Function (Sigmoid)**:
  $$f_{\mathbf{w}, b}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
  Includes numerical overflow protection with `np.clip(z, -500, 500)`.

- **Cost Function (Binary Cross-Entropy with L2 Regularization)**:
  $$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(f^{(i)}) + (1 - y^{(i)}) \log(1 - f^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2$$

- **Vectorized Gradient Updates**:
  $$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^T (\mathbf{f} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}$$
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (f^{(i)} - y^{(i)})$$

---

## 2. Model Comparison & Benchmarks

In diagnostic medicine, **Recall (Sensitivity)** on the malignant class is the critical metric: a False Negative (predicting benign when a tumor is malignant) can delay life-saving treatment, whereas a False Positive triggers secondary screening.

Models were evaluated on a held-out test split (20%, $n=114$) with standardized features:

| Model | Accuracy | Precision (Malignant) | Recall (Malignant) | F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 97.37% | 97.62% | 95.35% | 0.9647 | < 5 ms |
| **Random Forest** | 96.49% | 97.56% | 93.02% | 0.9524 | ~ 120 ms |
| **SVM (RBF Kernel)** | 97.37% | 97.62% | 95.35% | 0.9647 | ~ 8 ms |
| **Gradient Boosting** | 95.61% | 95.24% | 93.02% | 0.9412 | ~ 180 ms |
| **K-Nearest Neighbors** | 94.74% | 95.12% | 90.70% | 0.9286 | < 2 ms |

Logistic Regression and SVM delivered the strongest balance of high test accuracy (97.37%) and low inference latency.

---

## 3. Serving & Interfaces

### A. FastAPI REST Service (`src/api.py`)
Serves model inference over HTTP with automatic data validation via Pydantic:
- `POST /predict` — Accepts 30 feature values, returns diagnosis class and class probabilities.
- `GET /health` — Diagnostic check returning model status and validation metrics.
- `GET /feature-names` — Metadata endpoint returning feature ranges and training statistics.

Run the API:
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive OpenAPI/Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### B. Streamlit Diagnostic Dashboard (`app.py`)
An interactive dashboard allowing clinicians to adjust cell measurements across three categories:
- **Mean Features** (1–10)
- **Standard Error** (11–20)
- **Worst Values** (21–30)

Run the dashboard:
```bash
streamlit run app.py
```

---

## 4. Setup & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/BreastCancerPredictor.git
cd BreastCancerPredictor

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Benchmarks
To run the model comparisons and verify test metrics:
```bash
python3 src/compare_models.py
```

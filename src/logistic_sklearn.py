"""
logistic_sklearn.py
-------------------
Same Breast Cancer classification task, but using scikit-learn's
LogisticRegression instead of our hand-written gradient descent.

Goal: compare results side-by-side with logistic_from_scratch.py
      to verify our NumPy math is correct.
"""

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import get_data


def main():
    print("=" * 60)
    print("Logistic Regression with scikit-learn")
    print("Breast Cancer Diagnosis")
    print("=" * 60)

    # ── 1. Load data  ──
    X_train, X_test, y_train, y_test, feat_names, target_names = get_data()
    print(f"\nLoaded {X_train.shape[0]} training and {X_test.shape[0]} test patients, "
          f"{X_train.shape[1]} features each.")

    # ── 2. Scale features ──
   
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   
    X_test_s = scaler.transform(X_test)          
    print("Applied z-score normalization (StandardScaler).")

    # ── 3. Train ──
    
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",       
        random_state=42,
    )
    model.fit(X_train_s, y_train)
    print("Model trained!\n")

    # ── 4. Predict ──
    train_pred = model.predict(X_train_s)
    test_pred = model.predict(X_test_s)

    # ── 5. Evaluate ──
    print(f"Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test accuracy:     {accuracy_score(y_test, test_pred):.4f}")

    print("\nClassification Report (test set):")
    print(classification_report(y_test, test_pred, target_names=target_names))

    print("Confusion Matrix (test set):")
    cm = confusion_matrix(y_test, test_pred)
    print(f"  {target_names[0]:>10s}  {target_names[1]:>10s}   ← predicted")
    for i, row in enumerate(cm):
        print(f"  {row[0]:10d}  {row[1]:10d}   | actual {target_names[i]}")

    # ── 6. Show one example prediction ──
    idx = 5
    prob = model.predict_proba(X_test_s[idx].reshape(1, -1))[0]
    pred_label = target_names[model.predict(X_test_s[idx].reshape(1, -1))[0]]
    true_label = target_names[y_test[idx]]
    print(f"\nExample patient #{idx}:")
    print(f"  P(malignant) = {prob[0]:.3f}   P(benign) = {prob[1]:.3f}")
    print(f"  Predicted: {pred_label}   |   Actual: {true_label}")

    # ── 7. Feature importance (which measurements matter most?) ──
    print("\nTop 10 Most Important Features (by |weight|):")
    weights = model.coef_[0]
    top_idx = np.argsort(np.abs(weights))[::-1][:10]
    for rank, i in enumerate(top_idx, 1):
        direction = "→ benign" if weights[i] > 0 else "→ malignant"
        print(f"  {rank:2d}. {feat_names[i]:<28s}  w = {weights[i]:+.4f}  {direction}")


if __name__ == "__main__":
    main()

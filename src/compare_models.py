"""
compare_models.py
-----------------
Train multiple ML classifiers on the Breast Cancer dataset and
compare their performance side-by-side.

Models compared:
  1. Logistic Regression  (linear, what we already built)
  2. Random Forest        (ensemble of decision trees)
  3. Support Vector Machine (SVM)
  4. K-Nearest Neighbors  (KNN)
  5. Gradient Boosting    (sequential ensemble)
"""

import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import get_data


# ── Define models to compare ──
MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "SVM (RBF kernel)": SVC(
        kernel="rbf", C=1.0, random_state=42
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train a model and return a dict of metrics."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    # For cancer: malignant = 0 is the dangerous class.
    # We want high recall on malignant (don't miss cancer!)
    tp, tn, fp, fn = 0, 0, 0, 0
    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "train_acc": accuracy_score(y_train, train_pred),
        "test_acc": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, pos_label=0),
        "recall": recall_score(y_test, test_pred, pos_label=0),
        "f1": f1_score(y_test, test_pred, pos_label=0),
        "false_positives": fp,   # malignant called benign (risky!)
        "false_negatives": fn,   # benign called malignant
        "train_time_ms": train_time * 1000,
    }


def main():
    print("=" * 70)
    print("   Model Comparison — Breast Cancer Diagnosis")
    print("=" * 70)

    # ── Load & scale ──
    X_train, X_test, y_train, y_test, feat_names, target_names = get_data()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\nDataset: {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features")
    print(f"Classes: {target_names}\n")

    # ── Train & evaluate each model ──
    results = {}
    for name, model in MODELS.items():
        print(f"Training {name}...")
        results[name] = evaluate_model(model, X_train_s, X_test_s, y_train, y_test)

    # ── Results table ──
    print("\n" + "=" * 70)
    print(f"{'Model':<24s} {'Train':>7s} {'Test':>7s} {'Prec':>7s} "
          f"{'Recall':>7s} {'F1':>7s} {'FP':>4s} {'FN':>4s} {'Time':>8s}")
    print("-" * 70)

    # Sort by test accuracy (best first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["test_acc"], reverse=True)

    for rank, (name, m) in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal}{name:<22s} {m['train_acc']:>6.2%} {m['test_acc']:>6.2%} "
              f"{m['precision']:>6.2%} {m['recall']:>6.2%} {m['f1']:>6.2%} "
              f"{m['false_positives']:>4d} {m['false_negatives']:>4d} "
              f"{m['train_time_ms']:>6.1f}ms")

    print("-" * 70)

    # ── Winner summary ──
    best_name, best_metrics = sorted_results[0]
    print(f"\n Best model by test accuracy: {best_name} ({best_metrics['test_acc']:.2%})")

    # Also check which has best malignant recall (most important for cancer!)
    best_recall_name = max(results.items(), key=lambda x: x[1]["recall"])[0]
    best_recall_val = results[best_recall_name]["recall"]
    print(f"🔬 Best malignant recall (catching cancer): {best_recall_name} ({best_recall_val:.2%})")

    # Check for overfitting (big gap between train and test accuracy)
    print("\n  Overfitting check (train_acc - test_acc):")
    for name, m in sorted_results:
        gap = m["train_acc"] - m["test_acc"]
        flag = "   possible overfit!" if gap > 0.03 else ""
        print(f"   {name:<24s}  gap = {gap:.2%}{flag}")


if __name__ == "__main__":
    main()

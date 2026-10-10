import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import get_data
from src.utils import (
    fit_scaler, apply_scaler, accuracy, print_report, plot_cost_history,
)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost(X, y,w, b, lambda_ = 0.0):
    m = X.shape[0]
    f = sigmoid(X @ w + b)
    eps = 1e-15

    f = np.clip(f, eps, 1-eps)
    cost = -np.mean(y * np.log(f) + (1-y) *np.log(1-f))
    reg = (lambda_ / (2*m))*np.sum(w**2)
    return cost+reg

def compute_gradient(X, y, w, b, lambda_=0.0):
    m = X.shape[0]
    f = sigmoid(X @ w + b)
    error = f-y
    dj_dw = (X.T @ error) / m
    dj_db = np.mean(error)
    dj_dw += (lambda_/m) * w
    return dj_dw, dj_db

def gradient_descent(X, y, alpha=0.1, num_iters=2000,lambda_=0.0,verbose=True):
    m,n = X.shape
    w = np.zeros(n)
    b=0.0
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X,y,w,b,lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db 

        cost = compute_cost(X,y,w,b,lambda_)
        cost_history.append(cost)

        if verbose and (i% max(1,num_iters//10)==0 or i==num_iters-1):
            print(f" iters {i:5d} | cost {cost:.6f}")
    return w,b,cost_history

def predict(X,w,b,threshold=0.5):
    probs = sigmoid(X @ w + b)
    return (probs >= threshold).astype(int)

def main():
    print("=" * 60)
    print("Logistic Regression from scratch (NumPy only)")
    print("Breast Cancer Diagnosis")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feat_names, target_names = get_data()
    print(f"\nLoaded {X_train.shape[0]} training and {X_test.shape[0]} test patients,{X_train.shape[1]} features each.")

    mu,sigma = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, mu, sigma)
    X_test_s = apply_scaler(X_test,mu,sigma)
    print("Applied z-score normalization to features.")

    print("\nTraining with gradient descent...")
    w,b,cost_history = gradient_descent(X_train_s,y_train,alpha=0.1, num_iters=2000,lambda_=1.0)
    plot_cost_history(cost_history)
    train_pred = predict(X_train_s,w,b)
    test_pred = predict(X_test_s,w,b)
    print(f"Training accuracy: {accuracy(y_train,train_pred):.4f}")
    print(f"Test accuracy: {accuracy(y_test,test_pred):.4f}")
    print_report(y_test,test_pred, target_names=target_names)

    idx = 5
    prob = sigmoid(X_test_s[idx] @ w + b)
    pred_label = target_names[int(prob >= 0.5)]
    true_label = target_names[y_test[idx]]
    print(f"Example patient #{idx}:")
    print(f"  Model probability of 'benign': {prob:.3f}")
    print(f"  Predicted: {pred_label}   |   Actual: {true_label}")
   
if __name__ == "__main__":
    main()
    

import numpy as np

def fit_scaler(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    sigma[sigma == 0] = 1.0
    return mu,sigma

def apply_scaler(X,mu,sigma):
    return (X-mu)/sigma

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_counts(y_true, y_pred, positive=1):
    """Return (TP, TN, FP, FN) treating 'positive' as the positive class.

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    tn = np.sum((y_pred != positive) & (y_true != positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))

    return int(tp), int(tn), int(fp), int(fn)

def precision(y_true, y_pred, positive = 1):
    "Of everything we predicted positive, how many were actually positive?"
    tp,tn,fp,fn = confusion_counts(y_true,y_pred,positive)
    return tp/(tp+fp) if (tp+fp)>0 else 0.0

def recall(y_true, y_pred, positive=1):
    """Out of all the real positives, what fraction did we catch?"""
    tp,tn,fp,fn = confusion_counts(y_true, y_pred, positive)
    return tp/(tp+fn) if (tp+fn)>0 else 0.0

def f1_score(y_true, y_pred, positive=1):
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def print_report(y_true, y_pred, target_names=("malignant", "benign")):
    acc = accuracy(y_true, y_pred)
    print("\nEvaluation Report")
    print(f"Accuracy: {acc:.4f}")

    for pos, name in [(1, target_names[1]), (0, target_names[0])]:
        p = precision(y_true, y_pred, positive=pos)
        r = recall(y_true, y_pred, positive=pos)
        f = f1_score(y_true, y_pred, positive=pos)
        print(f"\n Treating '{name} (class {pos}) as positive:")
        print(f" Precision: {p:.4f}")
        print(f" Recall:    {r:.4f}")
        print(f" F1-score:  {f:.4f}")

    tp,tn,fp,fn = confusion_counts(y_true, y_pred, positive=1)
    print("\n  Confusion matrix (positive = benign / class 1):")
    print(f"    True Positives (benign correctly found):     {tp}")
    print(f"    True Negatives (malignant correctly found):  {tn}")
    print(f"    False Positives (malignant called benign):   {fp}  <- risky!")
    print(f"    False Negatives (benign called malignant):   {fn}\n")

def plot_cost_history(cost_history, save_path = None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7,4))
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w,b)")
    plt.title("Gradient descent: cost should decrease over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=120)
        print(f"Saved cost curve to {save_path}")
    else:
         plt.show()
            
    plt.close()
    
            



    
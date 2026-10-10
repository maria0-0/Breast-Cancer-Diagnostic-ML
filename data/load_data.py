"""
load_data.py
------------
Loads the Breast Cancer Wisconsin dataset that ships built-in with scikit-learn.
No internet download required.

The dataset:
  - 569 patients (examples)
  - 30 numeric features (mean radius, texture, perimeter, area, smoothness, ...)
  - Target: 0 = malignant (cancerous), 1 = benign (not cancerous)

  X  -> the input features  (m examples x n features)
  y  -> the output labels   (what we want to predict)
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def get_data(test_size=0.2, random_state=42):
    """
    Return train/test splits plus feature and target names.

    Returns
    -------
    X_train, X_test : np.ndarray
        Feature matrices (rows = patients, cols = measurements).
    y_train, y_test : np.ndarray
        Labels (0 = malignant, 1 = benign).
    feature_names : list[str]
    target_names  : list[str]   # ['malignant', 'benign']
    """
    dataset = load_breast_cancer()
    X = dataset.data            # shape (569, 30)
    y = dataset.target          # shape (569,)

    # Hold out a portion of the data to test on -- the model never sees it
    # during training. `stratify=y` keeps the same class balance in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        X_train, X_test,
        y_train, y_test,
        list(dataset.feature_names),
        list(dataset.target_names),
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, target_names = get_data()
    print("Dataset loaded successfully!")
    print(f"  Training examples: {X_train.shape[0]}")
    print(f"  Test examples:     {X_test.shape[0]}")
    print(f"  Features (n):      {X_train.shape[1]}")
    print(f"  Classes:           {target_names}")
    print(f"  First 5 features:  {feature_names[:5]}")

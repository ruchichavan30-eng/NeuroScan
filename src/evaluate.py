"""
evaluate.py
-----------
Evaluates a trained model on test data:
    - accuracy
    - confusion matrix (printed + saved as an image)
    - classification report (precision, recall, F1-score per class)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "models", "confusion_matrix.png")


def evaluate_model(model, X_test, y_test, save_plot=True):
    """
    Runs predictions on the test set and prints/plots evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # --- Accuracy ---
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # --- Classification report ---
    # Gives precision, recall, F1-score, and support (sample count)
    # for each class (0 = no tumor, 1 = tumor).
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

    # --- Confusion matrix ---
    # Rows = actual class, Columns = predicted class.
    # Diagonal = correct predictions, off-diagonal = mistakes.
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if save_plot:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Tumor", "Tumor"],
            yticklabels=["No Tumor", "Tumor"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        os.makedirs(os.path.dirname(CONF_MATRIX_PATH), exist_ok=True)
        plt.savefig(CONF_MATRIX_PATH)
        plt.close()
        print(f"Confusion matrix plot saved to: {CONF_MATRIX_PATH}")

    return acc

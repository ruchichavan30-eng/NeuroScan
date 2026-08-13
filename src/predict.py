"""
predict.py
----------
Loads the trained model + scaler and runs a prediction on a single
new MRI image.

Usage (command line):
    python src/predict.py path/to/image.jpg
"""

import os
import sys
import joblib

from preprocess import load_and_preprocess_image
from feature_extraction import extract_hog_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_hog_model.pkl")

LABEL_MAP = {0: "No Tumor", 1: "Tumor"}


def load_trained_model(model_path=MODEL_PATH):
    """Loads the saved dict containing the trained SVM and its scaler."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run src/train.py first."
        )
    saved = joblib.load(model_path)
    return saved["model"], saved["scaler"]


def predict_image(image_path, model=None, scaler=None):
    """
    Runs the full prediction pipeline on a single image:
    preprocess -> HOG features -> scale -> predict.

    Returns:
        dict with the predicted label, and the model's confidence
        (distance-based, since this SVM doesn't use probability=True
        by default - see note below).
    """
    if model is None or scaler is None:
        model, scaler = load_trained_model()

    processed_image = load_and_preprocess_image(image_path)
    if processed_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    features = extract_hog_features(processed_image)
    features_scaled = scaler.transform([features])  # scaler expects 2D input

    prediction = model.predict(features_scaled)[0]

    # decision_function gives a signed distance from the separating
    # boundary - further from 0 roughly means more confident.
    # (For calibrated probabilities instead, retrain SVC with
    # probability=True, which is slower to train.)
    decision_score = model.decision_function(features_scaled)[0]

    return {
        "label": LABEL_MAP[int(prediction)],
        "raw_prediction": int(prediction),
        "decision_score": float(decision_score),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_image(image_path)

    print(f"\nPrediction: {result['label']}")
    print(f"Decision score (confidence-ish, further from 0 = more confident): "
          f"{result['decision_score']:.3f}")
    print("\nNote: this is an academic image-classification result, "
          "NOT a medical diagnosis.")

"""
train.py
--------
Full training pipeline:
    load data -> extract HOG features -> train/test split
    -> scale features -> GridSearchCV over SVM params
    -> evaluate -> save the trained model

Run this file directly to train the model end-to-end:
    python src/train.py
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import load_dataset
from feature_extraction import extract_features_from_dataset
from evaluate import evaluate_model
from augmentation import augment_dataset

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_hog_model.pkl")

# ---- Hyperparameter grid for GridSearchCV ----
# GridSearchCV will try every combination below, using cross-validation,
# and keep whichever combination scores best on average.
# Trimmed down from our earlier wider grid to keep training time
# reasonable on large datasets - still covers a solid range of C values
# and both the most useful kernels.
PARAM_GRID = {
    "C": [0.1, 1, 10, 100],       # regularization strength
    "kernel": ["linear", "rbf"],  # decision boundary shape
    "gamma": ["scale"],           # used by 'rbf'
}


def train():
    # 1. Load and preprocess images
    print("Loading and preprocessing images...")
    images, labels = load_dataset(RAW_DATA_DIR)

    # 2. Train/test split - done on IMAGES first, before augmentation.
    # This is important: augmentation must only ever touch the training
    # portion. If we augmented before splitting, flipped/rotated copies
    # of the same original image could end up in both train and test,
    # letting the model "cheat" by having seen a near-duplicate already -
    # that would make our test accuracy misleadingly high.
    train_images, test_images, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train size: {len(train_images)}, Test size: {len(test_images)}")

    # 3. Augment ONLY the training images. For this large dataset, we
    # keep it to just a horizontal flip (rotation_angles=()) to keep
    # training time reasonable - see augmentation.py for details.
    train_images, y_train = augment_dataset(train_images, y_train, rotation_angles=())

    # 4. Extract HOG features (separately, so train/test never mix)
    print("Extracting HOG features...")
    X_train = extract_features_from_dataset(train_images)
    X_test = extract_features_from_dataset(test_images)

    # 5. Feature scaling
    # SVM is sensitive to feature scale. We fit the scaler ONLY on
    # training data, then apply it to both sets - this avoids leaking
    # information from the test set into training.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. GridSearchCV over the SVM
    # cv=5 means 5-fold cross-validation: the training data is split
    # into 5 parts, and each parameter combination is validated 5 times
    # to get a reliable performance estimate before picking the best one.
    #
    # class_weight='balanced' is important here: our dataset has far
    # more "Tumor" images than "No Tumor" images (since we collapsed 3
    # tumor categories into one class). Without this, the SVM could learn
    # to just favor predicting "Tumor" more often since it's the majority
    # class. class_weight='balanced' automatically gives more weight to
    # the minority class during training, correcting for this imbalance.
    # This is a standard, built-in scikit-learn SVM parameter - not new
    # technology, just using the tool correctly for imbalanced data.
    print("Running GridSearchCV (this may take a bit)...")
    grid_search = GridSearchCV(
        estimator=SVC(class_weight="balanced"),
        param_grid=PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # 7. Evaluate on the held-out test set
    evaluate_model(best_model, X_test_scaled, y_test)

    # 8. Save model + scaler together
    # We save both because any new image at prediction time must be
    # scaled with the SAME scaler that was fit on training data.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": best_model, "scaler": scaler}, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()

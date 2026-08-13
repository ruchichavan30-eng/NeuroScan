"""
feature_extraction.py
----------------------
Converts preprocessed grayscale images into HOG (Histogram of
Oriented Gradients) feature vectors.

What HOG actually does, in simple terms:
- It divides the image into small cells (e.g. 8x8 pixels).
- In each cell, it looks at the direction ("orientation") in which
  brightness changes the most sharply (i.e. edges/gradients).
- It builds a small histogram of these edge directions per cell.
- These histograms from every cell are combined into one long
  1D vector, which numerically describes the image's shapes and
  textures - much more useful for a classifier than raw pixels.

This is exactly the technique used in the original project.
"""

import numpy as np
from skimage.feature import hog

# HOG hyperparameters - kept at commonly-used defaults for this
# type of grayscale medical image classification project.
HOG_PARAMS = {
    "orientations": 9,          # number of gradient direction bins
    "pixels_per_cell": (8, 8),  # size of each cell HOG looks at
    "cells_per_block": (2, 2),  # cells grouped together for normalization
    "block_norm": "L2-Hys",     # standard normalization method
}


def extract_hog_features(image, hog_params=HOG_PARAMS):
    """
    Extracts a HOG feature vector from a single preprocessed
    (grayscale, fixed-size) image.

    Returns:
        A 1D numpy array of HOG features.
    """
    features = hog(image, **hog_params)
    return features


def extract_features_from_dataset(images, hog_params=HOG_PARAMS):
    """
    Applies HOG extraction to a list of preprocessed images.

    Returns:
        A 2D numpy array of shape (num_images, num_hog_features),
        ready to be used as the X matrix for training.
    """
    feature_list = [extract_hog_features(img, hog_params) for img in images]
    return np.array(feature_list)


if __name__ == "__main__":
    # Quick manual test using the preprocessing module.
    import os
    from preprocess import load_dataset

    RAW_DATA_DIR = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw"
    )
    images, labels = load_dataset(RAW_DATA_DIR)
    X = extract_features_from_dataset(images)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {labels.shape}")

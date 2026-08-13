"""
augmentation.py
----------------
Creates extra training images by applying small, realistic
transformations to the existing ones - a classic technique for making
small datasets more robust, WITHOUT collecting new data or using any
new technology (still plain OpenCV/numpy operations).

Why this helps:
Our dataset only has 253 images. By adding horizontally-flipped and
slightly-rotated versions of each image, the SVM sees more variety of
"what a tumor can look like" (different orientation, slightly
different angle), which tends to make it generalize better to new,
unseen images instead of memorizing the exact 253 originals.

Important: augmentation is only applied to TRAINING data, never to
the test set - the test set must stay as real, original images so
our evaluation numbers reflect real-world performance.
"""

import cv2
import numpy as np


def flip_image(image):
    """Mirrors the image left-right. Anatomically reasonable for brain
    MRI slices, since tumors can appear on either hemisphere."""
    return cv2.flip(image, 1)


def rotate_image(image, angle):
    """Rotates the image by a small angle (in degrees) around its center."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def augment_dataset(images, labels, rotation_angles=(-10, 10)):
    """
    Takes the original training images/labels and returns an expanded
    set that includes the originals PLUS a horizontally flipped version
    of each.

    Note: for large datasets (thousands of images), we keep augmentation
    light - just 1 flip per image (roughly doubling the training set) -
    since the combination of a big dataset AND heavy augmentation AND a
    wide GridSearchCV search makes training impractically slow on a
    typical laptop CPU. For a small dataset (a few hundred images), you
    can pass rotation_angles to add more variety, since the base set is
    small enough that the extra compute cost stays manageable.

    Returns:
        augmented_images (list), augmented_labels (numpy array)
    """
    augmented_images = list(images)
    augmented_labels = list(labels)

    for image, label in zip(images, labels):
        augmented_images.append(flip_image(image))
        augmented_labels.append(label)

        for angle in rotation_angles:
            augmented_images.append(rotate_image(image, angle))
            augmented_labels.append(label)

    print(f"Augmentation: {len(images)} original images -> "
          f"{len(augmented_images)} images after augmentation")

    return augmented_images, np.array(augmented_labels)

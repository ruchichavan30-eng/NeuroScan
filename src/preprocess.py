"""
preprocess.py
--------------
Loads raw MRI images from data/raw/yes and data/raw/no,
converts them to grayscale, and resizes them to a fixed size.

Why this matters:
- HOG (our feature extractor) needs every image to be the SAME size,
  otherwise the resulting feature vectors won't have matching lengths.
- Grayscale is used because MRI scans don't carry meaningful color
  information (they're essentially intensity maps), and grayscale
  also keeps HOG's computation simpler and faster.
- CLAHE (contrast normalization) makes brightness/contrast consistent
  across images from different sources (different scanners, different
  photo quality) - without it, an image that's simply brighter or
  higher-contrast than the training images can "look different" to
  HOG even if it shows the same kind of tumor.
"""

import os
import cv2
import numpy as np

# Fixed size every image will be resized to (width, height).
# 128x128 is a common, safe choice: small enough to keep HOG fast,
# large enough to keep tumor-shape detail.
IMAGE_SIZE = (128, 128)

# CLAHE = Contrast Limited Adaptive Histogram Equalization.
# In simple terms: it boosts local contrast in each small region of the
# image so details are visible whether the original image was dark,
# bright, low-contrast, or high-contrast. This is a classical image
# processing technique (not a new ML model) - it's a standard OpenCV
# preprocessing step, very commonly used on medical images.
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_and_preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Loads a single image from disk and preprocesses it.

    Steps:
    1. Read the image from disk.
    2. Convert it to grayscale (removes color channels, keeps intensity).
    3. Resize to a fixed size so every image matches.
    4. Apply CLAHE to normalize contrast/brightness across images from
       different sources.

    Returns:
        A 2D numpy array (grayscale image), or None if the file
        couldn't be read (e.g. corrupted file).
    """
    image = cv2.imread(image_path)  # reads image in BGR color format

    if image is None:
        # cv2 returns None instead of raising an error on bad files,
        # so we check explicitly and skip such files.
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, image_size)
    normalized_image = _clahe.apply(resized_image)

    return normalized_image


def load_dataset(raw_data_dir, image_size=IMAGE_SIZE):
    """
    Walks through the dataset folder and preprocesses every image,
    relabeling into a BINARY tumor / no-tumor problem.

    Supports two dataset layouts:

    1. Simple binary layout:
        data/raw/yes/   -> tumor present   -> label 1
        data/raw/no/    -> tumor absent    -> label 0

    2. Bhuvaji-style 4-class layout (Training/Testing split, each with
       4 tumor-type subfolders). We COMBINE Training + Testing here and
       do our own train/test split later in train.py, so both parts are
       treated identically:
        data/raw/Training/glioma_tumor/
        data/raw/Training/meningioma_tumor/
        data/raw/Training/pituitary_tumor/
        data/raw/Training/no_tumor/
        data/raw/Testing/...  (same 4 subfolders)

       Any folder literally named "no_tumor" (or "no") -> label 0.
       Every other tumor-type folder (glioma/meningioma/pituitary) is
       collapsed into label 1, keeping this a binary "is there a
       tumor at all" project, matching the original scope.
    """
    images = []
    labels = []

    # Folder names that mean "no tumor" -> label 0.
    # Everything else found containing images is treated as "tumor" -> 1.
    NO_TUMOR_FOLDER_NAMES = {"no", "no_tumor", "notumor"}

    found_any_images = False

    # Walk every subfolder under raw_data_dir looking for image files.
    for root, _dirs, files in os.walk(raw_data_dir):
        folder_name = os.path.basename(root).lower()
        image_files = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            continue  # skip folders with no images (e.g. intermediate dirs)

        # Decide the label based on the folder name.
        if folder_name in NO_TUMOR_FOLDER_NAMES:
            label = 0
        elif folder_name == "yes":
            label = 1
        elif folder_name in {"glioma_tumor", "meningioma_tumor", "pituitary_tumor",
                              "glioma", "meningioma", "pituitary"}:
            label = 1
        else:
            # Unrecognized folder name - skip it rather than guess wrong.
            continue

        for filename in image_files:
            file_path = os.path.join(root, filename)
            processed = load_and_preprocess_image(file_path, image_size)
            if processed is not None:
                images.append(processed)
                labels.append(label)
                found_any_images = True
            else:
                print(f"Skipped unreadable file: {file_path}")

    if not found_any_images:
        raise FileNotFoundError(
            f"No images found under {raw_data_dir}.\n"
            f"Expected either data/raw/yes + data/raw/no, or a "
            f"Training/Testing folder structure with tumor-type subfolders."
        )

    print(f"Loaded {len(images)} images "
          f"({labels.count(1)} tumor, {labels.count(0)} no-tumor)")

    return images, np.array(labels)


if __name__ == "__main__":
    # Quick manual test: run this file directly to sanity-check loading.
    RAW_DATA_DIR = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw"
    )
    imgs, lbls = load_dataset(RAW_DATA_DIR)
    print(f"Example image shape: {imgs[0].shape if imgs else 'N/A'}")

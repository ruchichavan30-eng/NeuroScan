# NeuroScan (HOG-SVM) - Brain Tumor Detection

An academic machine learning project that classifies brain MRI images as
**Tumor** or **No Tumor**, using classical computer vision (HOG feature
extraction) and a Support Vector Machine (SVM) classifier tuned with
GridSearchCV.

> **Important:** This is an academic image-classification project, not a
> clinical or medically approved diagnostic system. Do not use it to make
> real medical decisions.

## Results

Trained on the ~3,264-image Bhuvaji dataset (see Dataset section), with
CLAHE contrast normalization, train-only data augmentation, and
`class_weight='balanced'` to correct for the tumor/no-tumor class
imbalance:

| Metric | Score |
|---|---|
| Cross-validation accuracy | 97.24% |
| Test accuracy (653 held-out images) | 96.17% |
| Tumor - precision / recall | 0.97 / 0.99 |
| No Tumor - precision / recall | 0.91 / 0.83 |

Best parameters found by GridSearchCV: `C=10, kernel='rbf', gamma='scale'`.

No Tumor is the harder class to get right (fewer training examples even
after balancing), which shows up as somewhat lower recall on that class.

## Pipeline

```
MRI Images -> Preprocessing (grayscale, resize, CLAHE contrast normalization)
   -> Train/Test Split -> Data Augmentation (train set only)
   -> HOG Feature Extraction -> Feature Scaling
   -> SVM + GridSearchCV -> Evaluation -> Save Model
   -> Predict on New Image (script or UI)
```

## Project Structure

```
neuroscan-hog-svm/
├── data/raw/               # Dataset goes here (see Dataset section below)
├── models/                  # Saved trained model + confusion matrix plot
├── src/
│   ├── preprocess.py          # Load, grayscale, resize, CLAHE normalization
│   ├── augmentation.py        # Flip/rotate training images (train-only)
│   ├── feature_extraction.py  # HOG feature extraction
│   ├── train.py                # Full training pipeline
│   ├── evaluate.py             # Metrics: accuracy, confusion matrix, report
│   └── predict.py              # Predict on a single new image
├── ui/app.py                # Streamlit interface (titled "NeuroScan")
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
cd neuroscan-hog-svm
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the Dataset

This project uses the Kaggle dataset:
**"Brain Tumor Classification (MRI)"** by Sartaj Bhuvaji
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

It contains roughly 3,264 images across 4 folders (glioma_tumor,
meningioma_tumor, pituitary_tumor, no_tumor), already split into
`Training` and `Testing`.

Download and extract it, then copy both the `Training` and `Testing`
folders directly into `data/raw/`, so you end up with:

```
data/raw/Training/glioma_tumor/
data/raw/Training/meningioma_tumor/
data/raw/Training/pituitary_tumor/
data/raw/Training/no_tumor/
data/raw/Testing/glioma_tumor/
data/raw/Testing/meningioma_tumor/
data/raw/Testing/pituitary_tumor/
data/raw/Testing/no_tumor/
```

`src/preprocess.py` automatically finds all of these, combines them, and
relabels the three tumor-type folders as **Tumor (1)** and `no_tumor` as
**No Tumor (0)** - keeping this a binary classification project.

(It also still supports the simpler `data/raw/yes` / `data/raw/no`
layout - e.g. the smaller 253-image "Brain MRI Images for Brain Tumor
Detection" dataset - if you ever want to switch back to a smaller
dataset instead.)

## 3. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess all images (grayscale, resize, CLAHE contrast
  normalization)
- Split into train/test sets (80/20, stratified) - done BEFORE
  augmentation, so the test set stays completely untouched and honest
- Augment the training set only (horizontal flip) to give the SVM a
  bit more variety to learn from, without making training impractically
  slow on a large dataset
- Extract HOG features
- Scale features
- Run GridSearchCV over a range of SVM parameters (C, kernel, gamma),
  using `class_weight='balanced'` to correct for the tumor/no-tumor
  class imbalance in this dataset
- Print accuracy, classification report, and confusion matrix
- Save the trained model to `models/svm_hog_model.pkl`
- Save a confusion matrix plot to `models/confusion_matrix.png`

Note: with the full ~3,264-image dataset, this takes a real amount of
time (roughly 10-25 minutes on a typical laptop CPU) since HOG feature
extraction and GridSearchCV both run on CPU, not GPU. Let it run
uninterrupted.

## 4. Predict on a New Image (command line)

```bash
python src/predict.py path/to/some_mri_image.jpg
```

## 5. Run the Simple UI

```bash
streamlit run ui/app.py
```

This opens a local browser tab (usually http://localhost:8501) where you
can upload an MRI image, click "Run Prediction," and see the result.

## Known Limitations (honest notes)

- This is a classical HOG+SVM approach, not deep learning - it will
  generalize less well than a CNN to MRI images from very different
  scanners, resolutions, or heavily edited/watermarked sources (e.g.
  stock photos with visible text overlays).
- Test accuracy naturally fluctuates somewhat between training runs.
- Best, most reliable results come from testing on real, unaltered MRI
  images similar in style to the training dataset.

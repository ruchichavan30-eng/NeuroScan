"""
app.py
------
A minimal Streamlit interface for the Brain Tumor Detection project.

What it does (only this - nothing more):
1. Lets the user upload an MRI image.
2. Runs the trained HOG+SVM model on it.
3. Displays the prediction result clearly.

Run with:
    streamlit run ui/app.py
"""

import os
import sys
import tempfile

import streamlit as st

# Allow importing from ../src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src")
sys.path.append(SRC_DIR)

from predict import load_trained_model, predict_image  # noqa: E402

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("NeuroScan")
st.caption(
    "Academic image-classification project - NOT a medical diagnostic tool. "
    "Do not use this for real medical decisions."
)

# Load the model once and cache it, so it's not reloaded on every interaction.
@st.cache_resource
def get_model():
    return load_trained_model()


try:
    model, scaler = get_model()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(str(e))

uploaded_file = st.file_uploader(
    "Upload a brain MRI image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI image", use_container_width=True)

    if model_loaded and st.button("Run Prediction"):
        # Save the uploaded file to a temp path so our existing
        # OpenCV-based preprocessing (which reads from disk) can use it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        with st.spinner("Running HOG feature extraction and SVM prediction..."):
            result = predict_image(tmp_path, model=model, scaler=scaler)

        os.remove(tmp_path)

        st.subheader("Result")
        if result["raw_prediction"] == 1:
            st.error(f"Prediction: **{result['label']}**")
        else:
            st.success(f"Prediction: **{result['label']}**")

        st.write(f"Decision score: `{result['decision_score']:.3f}`")
        st.caption(
            "Score is the distance from the SVM's decision boundary - "
            "further from 0 roughly indicates a stronger prediction, "
            "not a calibrated probability."
        )

        st.info(
            "Reminder: this is a machine learning academic project result, "
            "not a clinical or medically approved diagnosis."
        )

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import tempfile

# ===================== Page Config =====================

st.set_page_config(
    page_title="FakeProof - Deepfake Detection",
    layout="wide"
)

# ===================== Load Model =====================

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("model/deepfake_video_model.h5")

model = load_trained_model()

# ===================== Constants =====================

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# ===================== Feature Extractor =====================

@st.cache_resource
def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)

    return tf.keras.Model(inputs, outputs)

feature_extractor = build_feature_extractor()

# ===================== Video Processing =====================

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()

    return np.array(frames)

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)

        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0)

        frame_mask[i, :length] = 1

    return frame_features, frame_mask

# ===================== UI =====================

# Background Image Styling
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/static/ai_face.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>FAKEPROOF: DETECTS DEEPFAKE VIDEOS</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#f1c40f; font-weight:bold;'>AI vs AI: Fighting deception with detection</h3>", unsafe_allow_html=True)

st.write("")
st.write("")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown("### Upload Your Video")

    uploaded_file = st.file_uploader("", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Show video preview
        st.video(uploaded_file)

        # Detect button
        if st.button("Detect Deepfake"):
            with st.spinner("Analyzing video... Please wait"):
                frames = load_video(tfile.name)
                frame_features, frame_mask = prepare_single_video(frames)

                prediction = model.predict([frame_features, frame_mask], verbose=0)[0][0]

                if prediction >= 0.51:
                    result = "FAKE"
                    confidence = round(float(prediction), 2)
                    st.markdown(f"<h3 style='color:#e74c3c;'>Result: {result}</h3>", unsafe_allow_html=True)
                else:
                    result = "REAL"
                    confidence = round(1 - float(prediction), 2)
                    st.markdown(f"<h3 style='color:#2ecc71;'>Result: {result}</h3>", unsafe_allow_html=True)

                st.markdown(f"<h4>Confidence: {confidence}</h4>", unsafe_allow_html=True)

        os.unlink(tfile.name)

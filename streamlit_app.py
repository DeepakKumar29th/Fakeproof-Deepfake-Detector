import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import tempfile

# ---------------- Page Config ----------------
st.set_page_config(page_title="FakeProof - Deepfake Detection", layout="wide")

# ---------------- Load Model ----------------
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("model/deepfake_video_model.h5", compile=False)

model = load_trained_model()

# ---------------- Constants ----------------
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# ---------------- Feature Extractor ----------------
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

# ---------------- Video Utilities ----------------
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
        length = min(MAX_SEQ_LENGTH, batch.shape[0])
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0)
        frame_mask[i, :length] = 1

    return frame_features, frame_mask

# ---------------- CSS Styling ----------------
st.markdown("""
<style>
.stApp {
    background: url("https://raw.githubusercontent.com/DeepakKumar29th/Fakeproof-Deepfake-Detector/main/static/ai_face.png");
    background-size: cover;
    background-position: center right;
    background-repeat: no-repeat;
}

.main-title {
    text-align: center;
    color: white;
    font-size: 42px;
    font-weight: bold;
    margin-top: 40px;
}

.sub-title {
    text-align: center;
    color: #f1c40f;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 30px;
}

.upload-card {
    background-color: #f4f4f4;
    padding: 25px;
    border-radius: 14px;
    text-align: center;
    width: 480px;
    margin: auto;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}

.result-real {
    color: #2ecc71;
    font-size: 22px;
    font-weight: bold;
}

.result-fake {
    color: #e74c3c;
    font-size: 22px;
    font-weight: bold;
}

.conf-text {
    color: black;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Titles ----------------
st.markdown("<div class='main-title'>FAKEPROOF: DETECTS DEEPFAKE VIDEOS</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI vs AI: Fighting deception with detection</div>", unsafe_allow_html=True)

# ---------------- Upload Card ----------------
st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
st.markdown("<h3 style='color:black;'>Upload Your Video</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4","avi","mov"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("Detect Deepfake"):
        with st.spinner("Analyzing video... Please wait"):

            frames = load_video(temp_file.name)
            frame_features, frame_mask = prepare_single_video(frames)
            prediction = model.predict([frame_features, frame_mask], verbose=0)[0][0]

            os.unlink(temp_file.name)

            if prediction >= 0.51:
                st.markdown("<div class='result-fake'>Result: FAKE</div>", unsafe_allow_html=True)
                confidence = round(float(prediction), 2)
            else:
                st.markdown("<div class='result-real'>Result: REAL</div>", unsafe_allow_html=True)
                confidence = round(1 - float(prediction), 2)

            st.markdown(f"<div class='conf-text'>Confidence: {confidence}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

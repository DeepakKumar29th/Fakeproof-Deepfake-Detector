from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('./model/deepfake_video_model.h5')

# Define constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# ===================== Feature Extractor =====================

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

    return tf.keras.Model(inputs, outputs, name="feature_extractor")

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
            frame = frame[:, :, [2, 1, 0]]  # BGR → RGB
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()

    return np.array(frames)

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)

        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0)

        frame_mask[i, :length] = 1

    return frame_features, frame_mask

# ===================== Routes =====================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']

    if video.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    # Save uploaded video
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)

    # Process video
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)

    # Predict
    prediction = model.predict([frame_features, frame_mask], verbose=0)[0][0]

    # Apply label + rounded confidence
    if prediction >= 0.51:
        result = "FAKE"
        confidence = round(float(prediction), 2)
    else:
        result = "REAL"
        confidence = round(1 - float(prediction), 2)

    # Remove uploaded file after prediction
    os.remove(video_path)

    return jsonify({'result': result, 'confidence': confidence})

# ===================== Main =====================

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)

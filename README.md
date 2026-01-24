# FakeProof: DeepFake Video Detector

## Overview

**FakeProof** is a web-based application that detects whether an uploaded video is **REAL** or **DEEPFAKE**.  
The system uses a **pre-trained deep learning model** to analyze video frames and identify manipulated content through an interactive web interface.

This project is developed for **academic, research, and demonstration purposes** to showcase AI-based deepfake detection.

---

## Key Features

- Detects **Real vs DeepFake** videos  
- Upload and preview videos directly in the web app  
- Displays prediction result with confidence score  
- Clean and interactive user interface  
- Optimized deep learning inference pipeline  
- Ready for cloud deployment  

---

## Tech Stack

- **Framework:** Streamlit  
- **Programming Language:** Python  
- **Model:** Pre-trained Deep Learning Model (.h5)  
- **Libraries:** TensorFlow, OpenCV, NumPy  
- **Frontend UI:** Streamlit Components  
- **Deployment:** Streamlit Cloud  

---

## How It Works

1. User uploads a video file  
2. The system extracts key frames from the video  
3. Frames are preprocessed and passed to the trained model  
4. The model predicts **REAL** or **FAKE**  
5. The result and confidence score are displayed instantly  

---

## Model Highlights

- Frame-level feature extraction using a CNN backbone  
- Sequence-based video classification  
- Trained on labeled real and deepfake datasets  
- Optimized for efficient inference  

---

## Use Cases

- Media authenticity verification  
- Fake news detection  
- Digital forensics  
- Social media content moderation  
- Academic research and project demonstrations  

---

## Deployment

The application is deployed on **Streamlit Cloud** and can be accessed here:

🔗 **Live App:**  
https://fakeproof-deepfake-detector.streamlit.app/

The app runs online 24/7 and does not require local installation.

---

## Project Interface

### Web Interface

<img width="1916" height="931" alt="1" src="https://github.com/user-attachments/assets/1b79c39c-1c5b-4ae9-94fb-a81f2f224b20" />

---

### Detection Result – REAL

<img width="1918" height="928" alt="2" src="https://github.com/user-attachments/assets/52378a10-8eda-4600-b5b4-70fece359790" />

---

### Detection Result – FAKE

<img width="1918" height="923" alt="3" src="https://github.com/user-attachments/assets/216b8e61-2183-4cc4-8fd9-856828eda355" />

---





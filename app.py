import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import io
import base64
import os
import streamlit as st

def safe_image(path, width=None):
    """
    Safely display an image only if it exists.
    Returns True if displayed, False otherwise.
    """
    if os.path.exists(path):
        st.image(path, width=width)
        return True
    else:
        st.warning(f"Result file not found: {path}")
        return False
if "run" not in st.session_state:
    st.session_state.run = False


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide", page_title="Emotion & Sign Detection")

# ==============================
# HIDE STREAMLIT UI
# ==============================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================
# BACKGROUND IMAGE FUNCTION
# ==============================
def set_bg(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# SESSION STATE
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "run" not in st.session_state:
    st.session_state.run = False

# ==============================
# EMOTION MODEL
# ==============================
def build_emotion_model():
    model = Sequential([
        InputLayer(input_shape=(48, 48, 1)),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D((2,2)), Dropout(0.4),
        Conv2D(256, (3,3), activation="relu"),
        MaxPooling2D((2,2)), Dropout(0.4),
        Conv2D(512, (3,3), activation="relu"),
        MaxPooling2D((2,2)), Dropout(0.4),
        Conv2D(512, (3,3), activation="relu"),
        MaxPooling2D((2,2)), Dropout(0.4),
        Flatten(),
        Dense(512, activation="relu"), Dropout(0.4),
        Dense(256, activation="relu"), Dropout(0.3),
        Dense(7, activation="softmax")
    ])
    return model

# Load models
emotion_model = build_emotion_model()
emotion_model.load_weights("facialemotionmodel.h5")

sign_model = Sequential([
    InputLayer(input_shape=(63,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(5, activation="softmax")
])
sign_model.load_weights("sign_model.h5")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
sign_names = ["hello", "yes", "no", "thank_you", "welcome"]

# ==============================
# MEDIAPIPE
# ==============================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)
face_detector = mp_face.FaceDetection()

# ==============================
# HOME PAGE
# ==============================
if st.session_state.page == "home":
    set_bg("wallpapers/home.png")
    st.markdown("<h1 style='text-align:center;color:white;'>Emotion & Sign Detection</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3,1,3])
    with col2:
        if st.button("START"):
            st.session_state.page = "login"

# ==============================
# LOGIN PAGE
# ==============================
elif st.session_state.page == "login":
    set_bg("wallpapers/login.png")
    st.markdown("<h2 style='text-align:center;color:white;'>Login</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3,1,3])
    with col2:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if user and pwd:
                st.session_state.page = "detect"
                st.session_state.run = True
            else:
                st.warning("Enter credentials")

# ==============================
# DETECTION PAGE
# ==============================
elif st.session_state.page == "detect":
    set_bg("wallpapers/detect.png")
    st.markdown("<h2 style='text-align:center;'>Live Detection</h2>", unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 1])

    frame_window = left_col.image([])

    # ---- Right-side predictions (TOP) ----
    right_col.markdown("### 🔍 Predictions")
    emotion_text = right_col.empty()
    sign_text = right_col.empty()

    right_col.markdown("### 😊 Emotion Probabilities")
    emotion_bars = [right_col.progress(0, text=label) for label in emotion_labels]

    right_col.markdown("### ✋ Sign Probabilities")
    sign_bars = [right_col.progress(0, text=label) for label in sign_names]

    start_btn = right_col.button("▶ Start Detection")
    view_results = right_col.button("📊 View Results")

    if start_btn:
        st.session_state.run = True

    if st.session_state.run:
        cap = cv2.VideoCapture(0)
        emotion_result, sign_result = "Detecting...", "Detecting..."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------- EMOTION DETECTION ----------------
            face_res = face_detector.process(rgb)
            if face_res.detections:
                box = face_res.detections[0].location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, bh = int(box.width * w), int(box.height * h)

                if x >= 0 and y >= 0 and bw > 0 and bh > 0:
                    face = frame[y:y+bh, x:x+bw]
                    if face.size != 0:
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        face = cv2.resize(face, (48, 48)) / 255.0
                        face = face.reshape(1, 48, 48, 1)

                        pred = emotion_model.predict(face, verbose=0)[0]
                        emotion_result = emotion_labels[np.argmax(pred)]

                        for i, bar in enumerate(emotion_bars):
                            bar.progress(int(pred[i] * 100), text=emotion_labels[i])
            else:
                emotion_result = "No Face"
                for bar in emotion_bars:
                    bar.progress(0)

            # ---------------- SIGN DETECTION ----------------
            hand_res = hands.process(rgb)
            if hand_res.multi_hand_landmarks:
                hand = hand_res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                pred_sign = sign_model.predict(
                    np.array(landmarks).reshape(1, -1), verbose=0
                )[0]

                sign_result = sign_names[np.argmax(pred_sign)]

                for i, bar in enumerate(sign_bars):
                    bar.progress(int(pred_sign[i] * 100), text=sign_names[i])
            else:
                sign_result = "No Hand"
                for bar in sign_bars:
                    bar.progress(0)

            # ---------------- UI UPDATES ----------------
            emotion_text.markdown(f"**Emotion:** {emotion_result}")
            sign_text.markdown(f"**Sign:** {sign_result}")

            cv2.putText(frame, f"Emotion: {emotion_result}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Sign: {sign_result}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            frame_window.image(frame, channels="BGR")

            # ---- View Results Button ----
            if view_results:
                st.session_state.run = False
                st.session_state.emotion = emotion_result
                st.session_state.sign = sign_result
                st.session_state.page = "result"
                break

        cap.release()


## ==============================
# RESULT PAGE
# ==============================
elif st.session_state.page == "result":
    set_bg("wallpapers/result.png")

    st.markdown(
        "<h2 style='text-align:center;color:white;'>Final Results</h2>",
        unsafe_allow_html=True
    )

    # ------------------------------
    # SIGN MODEL PERFORMANCE
    # ------------------------------
    st.subheader("📊 Sign Language Model Performance")

    st.success("✔ Validation Accuracy: **99.8%**")
    st.info("✔ Precision: 0.99  ✔ Recall: 0.99  ✔ F1-Score: 0.99")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Confusion Matrix")
        if safe_image("results/sign_confusion_matrix.png", width=450):
            with open("results/sign_confusion_matrix.png", "rb") as f:
                st.download_button(
                    "⬇ Download Confusion Matrix",
                    f,
                    file_name="sign_confusion_matrix.png"
                )

    with col2:
        st.markdown("### ROC Curve")
        if safe_image("results/sign_roc_curve.png", width=450):
            with open("results/sign_roc_curve.png", "rb") as f:
                st.download_button(
                    "⬇ Download ROC Curve",
                    f,
                    file_name="sign_roc_curve.png"
                )

    st.markdown("---")

    
    # ------------------------------
    # NAVIGATION
    # ------------------------------
    col1, col2, col3 = st.columns([3,1,3])
    with col2:
        if st.button("🏠 HOME"):
            st.session_state.page = "home"

import cv2
import mediapipe as mp
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
SIGN_NAME = "welcome"   # change this for each sign
LABEL = 4            # must match label dictionary
SAMPLES = 300         # number of samples to collect

SAVE_DIR = "sign_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# MediaPipe Hands Setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -----------------------------
# Storage
# -----------------------------
X = []
y = []

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
count = 0

print(f"Collecting data for sign: {SIGN_NAME}")

while cap.isOpened() and count < SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        X.append(landmarks)
        y.append(LABEL)
        count += 1

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.putText(frame, f"Sign: {SIGN_NAME}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Samples: {count}/{SAMPLES}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Sign Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Save data
# -----------------------------
X = np.array(X)
y = np.array(y)

np.save(os.path.join(SAVE_DIR, f"X_{SIGN_NAME}.npy"), X)
np.save(os.path.join(SAVE_DIR, f"y_{SIGN_NAME}.npy"), y)

print("✅ Data collection complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)

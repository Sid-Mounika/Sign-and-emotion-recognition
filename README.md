🎯 AI-Based Emotion & Sign Language Detection System
🧠 Overview

This project is a real-time AI application that detects facial emotions and hand sign gestures using deep learning and computer vision. It uses a webcam to capture live video and provides instant predictions with an interactive interface.


🚀 Features
🎥 Real-time webcam detection
😊 Emotion recognition (7 classes)
✋ Sign language detection (5 gestures)
📊 Live probability visualization
🔐 Login interface
📈 Performance metrics (accuracy, confusion matrix, ROC curve)
🌐 User-friendly UI using Streamlit



🛠️ Technologies Used
Python
OpenCV
MediaPipe
TensorFlow / Keras
Streamlit
NumPy, Pandas
Matplotlib, Seaborn



🤖 Models
Emotion Detection

CNN-based model
Input: 48x48 grayscale image
Output: 7 emotions
Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
Sign Language Detection

Neural Network model
Input: 63 hand landmark features
Output: 5 signs
Hello, Yes, No, Thank You, Welcome
📊 Results
✅ Accuracy: 99.8% (Sign Detection)
✅ High precision, recall, and F1-score
📉 Includes confusion matrix and ROC curve



💡 Use Cases
Assistive technology for hearing/speech impaired
Human-computer interaction
Emotion-aware AI systems
Smart surveillance


▶️ How to Run
pip install -r requirements.txt
streamlit run app.py


📁 Project Structure
├── app.py
├── facialemotionmodel.h5
├── sign_model.h5
├── wallpapers/
├── results/
└── README.md


📸 Screenshots

<img width="1921" height="969" alt="image" src="https://github.com/user-attachments/assets/e80857b0-a8b6-46de-bcb6-571ebb4ddd3e" />
<img width="1922" height="957" alt="image" src="https://github.com/user-attachments/assets/2fea395a-6660-45ae-97f0-dd5930145e18" />
<img width="1916" height="971" alt="image" src="https://github.com/user-attachments/assets/c349db8f-2802-4475-926f-575336cccc50" />
<img width="1922" height="971" alt="image" src="https://github.com/user-attachments/assets/b1808f10-e53f-4c08-8dbc-8a32565a8e4f" />




👩‍💻 Author

Siddam Mounika
Python Developer | AI & Deep Learning Enthusiast
AI-based real-time Emotion and Sign Language Detection system using Deep Learning and Computer Vision. Built with TensorFlow, OpenCV, MediaPipe, and Streamlit. Detects facial emotions and hand gestures via webcam with live predictions, probability scores, and an interactive user interface.

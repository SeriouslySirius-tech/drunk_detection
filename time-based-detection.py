import cv2
import mediapipe as mp
import numpy as np
import math
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from facedetection import *
import time
import sys

os.chdir("/Users/pranavyadlapati/Desktop/MiniProject")
model = joblib.load("logistic_regression_model-v2.pkl")
scaler = joblib.load("scaler.pkl")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print("Press 'q' to quit the program.")
start = time.time()
while (time.time() - start) < 5 and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break
    # analyzed_frame = analyze_frame(frame)
    frame_copy = frame.copy()
    features = extract_image(frame)
    print(f"Current Features: {features}")
    if features is None:
        cv2.putText(
            frame,
            "No face detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Face Feature Analysis (Press 'q' to Quit)", frame)
        start = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    outlined_frame = outline_frame(frame)
    feature_array = np.array(list(features.values())).reshape(1, -1)
    feature_array = np.nan_to_num(feature_array, nan=0)
    scaled_features = scaler.transform(feature_array)
    prediction = model.predict(scaled_features)
    status = "Drunk" if prediction[0] == 1 else "Sober"
    cv2.putText(
        frame,
        f"Drunk: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Face Feature Analysis (Press 'q' to Quit)", frame)
    print(prediction)
    if prediction[0] == 1:
        # time.sleep(3)
        sys.exit(1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit(0)

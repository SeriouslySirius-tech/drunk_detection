import cv2
import mediapipe as mp
import numpy as np
import math
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from facedetection import *

os.chdir("/Users/pranavyadlapati/Desktop/MiniProject")
model = joblib.load("random_forest_model-v2.pkl")
scaler = joblib.load("scaler.pkl")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print("Press 'q' to quit the program.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break
    # analyzed_frame = analyze_frame(frame)
    outlined_frame = outline_frame(frame)
    features = extract_image(frame)
    if features is None:
        cv2.imshow("Face Feature Analysis (Press 'q' to Quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    feature_array = np.array(list(features.values())).reshape(1, -1)
    feature_array = np.nan_to_num(feature_array, nan=0)
    scaled_features = scaler.transform(feature_array)
    prediction = model.predict(scaled_features)

    cv2.putText(
        frame,
        f"Drunk: {prediction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Face Feature Analysis (Press 'q' to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

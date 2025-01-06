import cv2
import mediapipe as mp
import numpy as np
import math
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from facedetection import *

# Initialize the webcam (0 is the default camera; change to 1, 2, etc., for other cameras)
# cap = cv2.VideoCapture(0)
os.chdir("/Users/pranavyadlapati/Desktop/MiniProject")
model = joblib.load("logistic_regression_model-v2.pkl")
scaler = joblib.load("scaler.pkl")

frame = cv2.imread("someimage.jpeg")

outlined_frame = outline_frame(frame)
features = extract_image(frame)
feature_array = np.array(list(features.values())).reshape(1, -1)
feature_array = np.nan_to_num(feature_array, nan=0)
scaled_features = scaler.transform(feature_array)
prediction = "Drunk" if model.predict(scaled_features) == 1 else "Sober"
cv2.putText(
    frame,
    f"Drunk Status: {prediction}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2,
)
while True:
    cv2.imshow("Face Feature Analysis (Press 'q' to Quit and 'r' to Reset)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit the program if 'q' is pressed
        print("Quitting...")
        break


# if not cap.isOpened():
#     print("Error: Unable to access the webcam.")
#     exit()

# print("Press 's' to save the image, or 'q' to quit.")

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Unable to capture a frame from the webcam.")
#         break

#     # Display the frame
#     cv2.imshow("Webcam - Press 's' to Save", frame)

#     # Check for keypress
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("s"):  # Save the image if 's' is pressed
#         outlined_frame = outline_frame(frame)
#         features = extract_image(frame)
#         if features is None:
#             cv2.imshow("Face Feature Analysis (Press 'q' to Quit)", outlined_frame)
#             continue

#         feature_array = np.array(list(features.values())).reshape(1, -1)
#         feature_array = np.nan_to_num(feature_array, nan=0)
#         scaled_features = scaler.transform(feature_array)
#         prediction = "Drunk" if model.predict(scaled_features) == 1 else "Sober"

#         cv2.putText(
#             frame,
#             f"Drunk Status: {prediction}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2,
#         )
#         cv2.imshow("Face Feature Analysis (Press 'q' to Quit and 'r' to Reset)", frame)

#     elif key == ord("q"):  # Quit the program if 'q' is pressed
#         print("Quitting...")
#         break

# # Show the original feed unless an outlined frame is being displayed


# # Release the webcam and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

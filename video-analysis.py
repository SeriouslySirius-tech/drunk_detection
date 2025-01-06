import cv2
from facedetection import *

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot Read Frame")
        break
    analysed_frame = analyze_frame(frame)
    cv2.imshow("Frame Analysis:", analysed_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

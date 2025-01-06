import cv2
import mediapipe as mp
import numpy as np
import math
import os
import pandas as pd

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

# Define regions of interest (ROI)
LEFT_CHEEK = [50, 187, 123, 116, 117]
RIGHT_CHEEK = [280, 411, 352, 346, 347]
FOREHEAD = [10, 63, 105, 66, 107]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 61, 291, 81, 311]
LEFT_BROW = [107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 123, 124, 125]
RIGHT_BROW = [336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 352, 351, 350]
MOUTH_OUTER = [61, 291, 13, 14, 78, 308]
SMILE_INNER = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
    61,
]
SMILE_OUTER = [
    13,
    14,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    291,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]
MOUTH_CORNERS = [61, 291]
MOUTH_CENTER = 13
LOWER_TONGUE = np.array([160, 100, 100])  # Lower bound for pink/reddish hues
UPPER_TONGUE = np.array([180, 255, 255])


def extract_region(frame, landmarks, indices):
    points = np.array([landmarks[i] for i in indices], dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    region = cv2.bitwise_and(frame, frame, mask=mask)
    return region, mask


def mouth_curvature(landmarks):
    """Calculate curvature of the mouth based on corners and center."""
    corner_left = np.array(landmarks[MOUTH_CORNERS[0]])
    corner_right = np.array(landmarks[MOUTH_CORNERS[1]])
    center = np.array(landmarks[MOUTH_CENTER])

    # Calculate vertical distances of corners relative to the center
    left_diff = corner_left[1] - center[1]
    right_diff = corner_right[1] - center[1]
    return (left_diff + right_diff) / 2.0


def aspect_ratio(landmarks, indices):
    """Calculate aspect ratio for eyes or mouth."""
    p = np.array([landmarks[i] for i in indices])
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_corner_curvature(landmarks):
    """Calculates curvature based on mouth corner positions."""
    try:
        left_corner = np.array(landmarks[61])
        right_corner = np.array(landmarks[291])
        upper_lip_center = np.array(landmarks[13])
        lower_lip_center = np.array(landmarks[14])

        # Calculate vectors
        v1 = upper_lip_center - left_corner
        v2 = lower_lip_center - left_corner
        v3 = upper_lip_center - right_corner
        v4 = lower_lip_center - right_corner

        # Calculate angles (using dot product)
        angle_left = np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )
        angle_right = np.arccos(
            np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4))
        )
        curvature = angle_left + angle_right
        return curvature
    except IndexError:
        return None


def calculate_redness(region):
    if np.count_nonzero(region) == 0:
        return 0.0
    b, g, r = cv2.split(region)
    avg_r = np.sum(r) / np.count_nonzero(region)
    avg_g = np.sum(g) / np.count_nonzero(region)
    avg_b = np.sum(b) / np.count_nonzero(region)
    redness_score = avg_r / (avg_g + avg_b + 1e-5)
    return redness_score


def aspect_ratio(landmarks, indices):
    p = np.array([landmarks[i] for i in indices])
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_brow_distance(landmarks, indices):
    p = np.array([landmarks[i] for i in indices])
    distances = []
    for i in range(1, len(p)):
        distances.append(np.linalg.norm(p[i] - p[i - 1]))
    return np.mean(distances) if distances else 0


def distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def mouth_openness(landmarks):
    """Calculate mouth openness based on upper and lower lip distance."""
    return distance(landmarks[13], landmarks[14])


"""
Forehead Occulsion Extraction
"""


def is_forehead_occluded(forehead_region):
    gray = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
    mean_intensity = cv2.mean(gray, mask=(gray > 0).astype(np.uint8))[
        0
    ]  # Mean intensity
    if (
        mean_intensity < 50 or mean_intensity > 200
    ):  # Thresholds (adjust based on lighting)
        return True  # Likely occluded
    return False


def detect_edges_in_forehead(forehead_region):
    gray = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Canny edge detection
    edge_density = np.sum(edges) / (
        np.count_nonzero(gray) + 1e-5
    )  # Normalize by non-black pixels
    return edge_density > 0.1  # Threshold for edge density


def eye_openness(landmarks, eye_indices):
    """Calculate average eye openness for given eye landmarks."""
    top = distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    bottom = distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    return (top + bottom) / 2


def calculate_emotion_scores(landmarks):
    """Calculate confidence scores for various emotions."""
    mouth_open = mouth_openness(landmarks)
    left_eye_open = eye_openness(landmarks, LEFT_EYE)
    right_eye_open = eye_openness(landmarks, RIGHT_EYE)
    avg_eye_open = (left_eye_open + right_eye_open) / 2

    # Define thresholds (empirical values, can be adjusted)
    max_mouth_open = 20  # Maximum observed mouth openness for happiness/surprise
    max_eye_open = 15  # Maximum observed eye openness for surprise

    # Normalize features (0 to 1 scale)
    mouth_score = min(mouth_open / max_mouth_open, 1.0)
    eye_score = min(avg_eye_open / max_eye_open, 1.0)
    left_eye_ar = aspect_ratio(landmarks, LEFT_EYE)
    right_eye_ar = aspect_ratio(landmarks, RIGHT_EYE)

    mouth_curve = mouth_curvature(landmarks)

    # Define emotion confidence scores
    scores = {
        "Happiness": (mouth_score * 0.6 + (1 - eye_score) * 0.4)
        * 100,  # Wide mouth, relaxed eyes
        "Surprise": (mouth_score * 0.5 + eye_score * 0.5)
        * 100,  # Wide mouth and wide eyes
        "Anger": ((1 - eye_score) * 0.4 + (1 - mouth_score) * 0.3)
        * 100,  # Squinted eyes, tight mouth
        "Neutral": (
            (1 - abs(mouth_score - 0.2)) * 0.5 + (1 - abs(eye_score - 0.5)) * 0.5
        )
        * 100,  # Balanced
        "Sadness": max(
            0, (mouth_curve / 20.0) + (0.5 - (left_eye_ar + right_eye_ar) / 2.0)
        ),
    }
    return scores


def calculate_head_orientation(landmarks, width, height):
    # Extract key landmarks
    left_eye = landmarks[33]  # Left eye
    right_eye = landmarks[263]  # Right eye
    nose_tip = landmarks[1]  # Nose tip
    chin = landmarks[199]  # Chin

    # Convert normalized landmarks to pixel coordinates
    def to_pixel(landmark):
        return int(landmark.x * width), int(landmark.y * height), landmark.z

    left_eye = to_pixel(left_eye)
    right_eye = to_pixel(right_eye)
    nose_tip = to_pixel(nose_tip)
    chin = to_pixel(chin)

    # Head Tilt (Roll)
    eye_slope = np.arctan2((right_eye[1] - left_eye[1]), (right_eye[0] - left_eye[0]))
    roll_angle = np.degrees(eye_slope)

    # Head Yaw
    mid_point_x = (left_eye[0] + right_eye[0]) / 2
    yaw_angle = np.degrees(np.arctan2(nose_tip[0] - mid_point_x, nose_tip[2]))

    # Head Pitch
    pitch_angle = np.degrees(np.arctan2(nose_tip[1] - chin[1], nose_tip[2] - chin[2]))

    return roll_angle, yaw_angle, pitch_angle


def is_tongue_out(landmarks, width, height):
    # Convert normalized landmarks to pixel coordinates
    # Convert landmarks to pixel coordinates
    def to_pixel(landmark):
        return int(landmark.x * width), int(landmark.y * height)

    upper_inner_lip = to_pixel(landmarks[13])
    lower_inner_lip = to_pixel(landmarks[14])
    left_corner_lip = to_pixel(landmarks[61])
    right_corner_lip = to_pixel(landmarks[291])

    vertical_distance = abs(lower_inner_lip[1] - upper_inner_lip[1])
    horizontal_distance = abs(right_corner_lip[0] - left_corner_lip[0])

    # Adjust thresholds here
    if horizontal_distance == 0:
        return False
    vertical_to_horizontal_ratio = vertical_distance / horizontal_distance

    min_vertical_distance = 15  # Minimum distance in pixels
    tongue_threshold_ratio = 0.2  # Adjust for sensitivity

    # Combine ratio and minimum distance checks
    if (
        vertical_distance > min_vertical_distance
        and vertical_to_horizontal_ratio > tongue_threshold_ratio
    ):
        return True
    return False


def analyze_frame(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [
                (int(l.x * frame.shape[1]), int(l.y * frame.shape[0]))
                for l in face_landmarks.landmark
            ]

            # Redness Analysis
            left_cheek_region, _ = extract_region(frame, landmarks, LEFT_CHEEK)
            right_cheek_region, _ = extract_region(frame, landmarks, RIGHT_CHEEK)
            forehead_region, _ = extract_region(frame, landmarks, FOREHEAD)
            left_eye_region, _ = extract_region(frame, landmarks, LEFT_EYE)
            right_eye_region, _ = extract_region(frame, landmarks, RIGHT_EYE)

            left_cheek_redness = calculate_redness(left_cheek_region)
            right_cheek_redness = calculate_redness(right_cheek_region)
            forehead_redness = calculate_redness(forehead_region)
            left_eye_redness = calculate_redness(left_eye_region)
            right_eye_redness = calculate_redness(right_eye_region)

            left_eye_ar = aspect_ratio(landmarks, LEFT_EYE)
            right_eye_ar = aspect_ratio(landmarks, RIGHT_EYE)
            mouth_ar = aspect_ratio(landmarks, MOUTH)

            left_brow_distance = calculate_brow_distance(landmarks, LEFT_BROW)
            right_brow_distance = calculate_brow_distance(landmarks, RIGHT_BROW)

            emotion_scores = calculate_emotion_scores(landmarks)
            forehead_occluded = is_forehead_occluded(
                forehead_region
            ) or detect_edges_in_forehead(forehead_region)
            mar = aspect_ratio(landmarks, MOUTH)
            curvature = mouth_corner_curvature(landmarks)

            height, width, _ = frame.shape
            roll, yaw, pitch = calculate_head_orientation(
                face_landmarks.landmark, width, height
            )
            tongue_out = is_tongue_out(face_landmarks.landmark, width, height)
            # Display Text
            cv2.putText(
                frame,
                f"L Cheek Red: {left_cheek_redness:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"R Cheek Red: {right_cheek_redness:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Forehead Red: {forehead_redness:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"L Eye Red: {left_eye_redness:.2f}",
                (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"R Eye Red: {right_eye_redness:.2f}",
                (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Left Eye AR: {left_eye_ar:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Right Eye AR: {right_eye_ar:.2f}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Mouth AR: {mouth_ar:.2f}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"L Brow Dist: {left_brow_distance:.2f}",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"R Brow Dist: {right_brow_distance:.2f}",
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Anger: {emotion_scores['Anger']:.2f}",
                (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Surprise: {emotion_scores['Surprise']:.2f}",
                (10, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Neutral: {emotion_scores['Neutral']:.2f}",
                (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Happiness: {emotion_scores['Happiness']:.2f}",
                (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Sadness: {emotion_scores['Sadness']:.2f}",
                (10, 480),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

            if mar is not None and curvature is not None:
                smile_metric = (
                    curvature / mar
                )  # This is the most important part, the smile metric is the curvature divided by the mouth aspect ratio.
                cv2.putText(
                    frame,
                    f"Smile Metric: {smile_metric:.2f}",
                    (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Forehead Oclcuded: {forehead_occluded}",
                (10, 510),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Roll: {roll:.2f} degrees",
                (10, 540),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Yaw: {yaw:.2f} degrees",
                (10, 570),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Pitch: {pitch:.2f} degrees",
                (10, 600),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Tongue Out: {tongue_out}",
                (10, 630),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Draw Landmarks
            for region, color in [
                (LEFT_EYE, (255, 0, 0)),
                (RIGHT_EYE, (255, 0, 0)),
                (MOUTH, (0, 0, 255)),
                (LEFT_BROW, (0, 165, 255)),
                (RIGHT_BROW, (0, 165, 255)),
                (SMILE_OUTER, (0, 0, 255)),
                (SMILE_INNER, (0, 0, 255)),
                (UPPER_TONGUE, (0, 0, 255)),
                (LOWER_TONGUE, (0, 0, 255)),
                (LEFT_CHEEK, (255, 255, 0)),
                (RIGHT_CHEEK, (255, 255, 0)),
                (FOREHEAD, (255, 255, 0)),
            ]:
                points = np.array([landmarks[i] for i in region], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=1)

            return frame
    else:
        cv2.putText(
            frame,
            "No face detected.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        return frame


def outline_frame(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [
                (int(l.x * frame.shape[1]), int(l.y * frame.shape[0]))
                for l in face_landmarks.landmark
            ]

            # Redness Analysis
            left_cheek_region, _ = extract_region(frame, landmarks, LEFT_CHEEK)
            right_cheek_region, _ = extract_region(frame, landmarks, RIGHT_CHEEK)
            forehead_region, _ = extract_region(frame, landmarks, FOREHEAD)
            left_eye_region, _ = extract_region(frame, landmarks, LEFT_EYE)
            right_eye_region, _ = extract_region(frame, landmarks, RIGHT_EYE)

            left_cheek_redness = calculate_redness(left_cheek_region)
            right_cheek_redness = calculate_redness(right_cheek_region)
            forehead_redness = calculate_redness(forehead_region)
            left_eye_redness = calculate_redness(left_eye_region)
            right_eye_redness = calculate_redness(right_eye_region)

            left_eye_ar = aspect_ratio(landmarks, LEFT_EYE)
            right_eye_ar = aspect_ratio(landmarks, RIGHT_EYE)
            mouth_ar = aspect_ratio(landmarks, MOUTH)

            left_brow_distance = calculate_brow_distance(landmarks, LEFT_BROW)
            right_brow_distance = calculate_brow_distance(landmarks, RIGHT_BROW)

            emotion_scores = calculate_emotion_scores(landmarks)
            forehead_occluded = is_forehead_occluded(
                forehead_region
            ) or detect_edges_in_forehead(forehead_region)
            mar = aspect_ratio(landmarks, MOUTH)
            curvature = mouth_corner_curvature(landmarks)

            height, width, _ = frame.shape
            roll, yaw, pitch = calculate_head_orientation(
                face_landmarks.landmark, width, height
            )
            tongue_out = is_tongue_out(face_landmarks.landmark, width, height)
            # Display Text

            # Draw Landmarks
            for region, color in [
                (LEFT_EYE, (255, 0, 0)),
                (RIGHT_EYE, (255, 0, 0)),
                (MOUTH, (0, 0, 255)),
                (LEFT_BROW, (0, 165, 255)),
                (RIGHT_BROW, (0, 165, 255)),
                (SMILE_OUTER, (0, 0, 255)),
                (SMILE_INNER, (0, 0, 255)),
                (UPPER_TONGUE, (0, 0, 255)),
                (LOWER_TONGUE, (0, 0, 255)),
                (LEFT_CHEEK, (255, 255, 0)),
                (RIGHT_CHEEK, (255, 255, 0)),
                (FOREHEAD, (255, 255, 0)),
            ]:
                points = np.array([landmarks[i] for i in region], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=1)

            return frame
    else:
        cv2.putText(
            frame,
            "No face detected.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        return frame


def extract_image(frame):
    try:
        features = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except cv2.error as e:
        print(f"OpenCV error during color conversion: {e}")
        return None
    if features.multi_face_landmarks:
        for face_landmarks in features.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [
                (int(l.x * frame.shape[1]), int(l.y * frame.shape[0]))
                for l in face_landmarks.landmark
            ]

            face_boundary_indices = list(
                range(0, 468)
            )  # Indices for all landmarks (adjust if needed)
            face_boundary_points = np.array(
                [landmarks[i] for i in face_boundary_indices], dtype=np.int32
            )
            hull = cv2.convexHull(face_boundary_points)

            # Create a mask for the face
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [hull], 255)

            # Extract the face region
            face_region = cv2.bitwise_and(frame, frame, mask=mask)

            # Optionally, crop the face region to a bounding rectangle
            x, y, w, h = cv2.boundingRect(hull)
            frame = face_region[y : y + h, x : x + w]
            try:
                frame = cv2.resize(frame, (256, 256))
            except cv2.error:
                return None

            # Redness Analysis
            left_cheek_region, _ = extract_region(frame, landmarks, LEFT_CHEEK)
            right_cheek_region, _ = extract_region(frame, landmarks, RIGHT_CHEEK)
            forehead_region, _ = extract_region(frame, landmarks, FOREHEAD)
            left_eye_region, _ = extract_region(frame, landmarks, LEFT_EYE)
            right_eye_region, _ = extract_region(frame, landmarks, RIGHT_EYE)

            left_cheek_redness = calculate_redness(left_cheek_region)
            right_cheek_redness = calculate_redness(right_cheek_region)
            forehead_redness = calculate_redness(forehead_region)
            left_eye_redness = calculate_redness(left_eye_region)
            right_eye_redness = calculate_redness(right_eye_region)

            left_eye_ar = aspect_ratio(landmarks, LEFT_EYE)
            right_eye_ar = aspect_ratio(landmarks, RIGHT_EYE)
            mouth_ar = aspect_ratio(landmarks, MOUTH)

            left_brow_distance = calculate_brow_distance(landmarks, LEFT_BROW)
            right_brow_distance = calculate_brow_distance(landmarks, RIGHT_BROW)

            emotion_scores = calculate_emotion_scores(landmarks)
            forehead_occluded = is_forehead_occluded(
                forehead_region
            ) or detect_edges_in_forehead(forehead_region)
            mar = aspect_ratio(landmarks, MOUTH)
            curvature = mouth_corner_curvature(landmarks)

            height, width, _ = frame.shape
            roll, yaw, pitch = calculate_head_orientation(
                face_landmarks.landmark, width, height
            )
            tongue_out = is_tongue_out(face_landmarks.landmark, width, height)
            features = {}
            features["LCheekRed"] = left_cheek_redness
            features["RCheekRed"] = right_cheek_redness
            features["LeftEyeAR"] = left_eye_ar
            features["RightEyeAR"] = right_eye_ar
            features["ForeheadRed"] = forehead_redness
            features["LEyeRedness"] = left_eye_redness
            features["REyeRedness"] = right_eye_redness
            features["MouthAR"] = mouth_ar
            features["LBrowDist"] = left_brow_distance
            features["RBrowDist"] = right_brow_distance
            features["Anger"] = emotion_scores["Anger"]
            features["Surprise"] = emotion_scores["Surprise"]
            features["Neutral"] = emotion_scores["Happiness"]
            features["Sadness"] = emotion_scores["Sadness"]
            if mar is not None and curvature is not None:
                smile_metric = curvature / mar
                features["Smile"] = smile_metric
            else:
                features["Smile"] = 0
            features["ForeheadOccluded"] = 0 if forehead_occluded == False else 1
            features["Roll"] = roll
            features["Pitch"] = pitch
            features["Yaw"] = yaw
            features["Tongue"] = 0 if tongue_out == False else 1
        return features
    else:
        return None


def extract_image1(frame):
    try:
        features = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except cv2.error as e:
        print(f"OpenCV error during color conversion: {e}")
        return None
    if features.multi_face_landmarks:
        for face_landmarks in features.multi_face_landmarks:
            # h, w, _ = frame.shape
            landmarks = [
                (int(l.x * frame.shape[1]), int(l.y * frame.shape[0]))
                for l in face_landmarks.landmark
            ]

            # face_boundary_indices = list(
            #     range(0, 468)
            # )  # Indices for all landmarks (adjust if needed)
            # face_boundary_points = np.array(
            #     [landmarks[i] for i in face_boundary_indices], dtype=np.int32
            # )
            # hull = cv2.convexHull(face_boundary_points)

            # # Create a mask for the face
            # mask = np.zeros((h, w), dtype=np.uint8)
            # cv2.fillPoly(mask, [hull], 255)

            # # Extract the face region
            # face_region = cv2.bitwise_and(frame, frame, mask=mask)

            # # Optionally, crop the face region to a bounding rectangle
            # x, y, w, h = cv2.boundingRect(hull)
            # frame = face_region[y : y + h, x : x + w]
            # try:
            #     frame = cv2.resize(frame, (256, 256))
            # except cv2.error:
            #     return None

            # Redness Analysis
            left_cheek_region, _ = extract_region(frame, landmarks, LEFT_CHEEK)
            right_cheek_region, _ = extract_region(frame, landmarks, RIGHT_CHEEK)
            forehead_region, _ = extract_region(frame, landmarks, FOREHEAD)
            left_eye_region, _ = extract_region(frame, landmarks, LEFT_EYE)
            right_eye_region, _ = extract_region(frame, landmarks, RIGHT_EYE)

            left_cheek_redness = calculate_redness(left_cheek_region)
            right_cheek_redness = calculate_redness(right_cheek_region)
            forehead_redness = calculate_redness(forehead_region)
            left_eye_redness = calculate_redness(left_eye_region)
            right_eye_redness = calculate_redness(right_eye_region)

            left_eye_ar = aspect_ratio(landmarks, LEFT_EYE)
            right_eye_ar = aspect_ratio(landmarks, RIGHT_EYE)
            mouth_ar = aspect_ratio(landmarks, MOUTH)

            left_brow_distance = calculate_brow_distance(landmarks, LEFT_BROW)
            right_brow_distance = calculate_brow_distance(landmarks, RIGHT_BROW)

            emotion_scores = calculate_emotion_scores(landmarks)
            forehead_occluded = is_forehead_occluded(
                forehead_region
            ) or detect_edges_in_forehead(forehead_region)
            mar = aspect_ratio(landmarks, MOUTH)
            curvature = mouth_corner_curvature(landmarks)

            height, width, _ = frame.shape
            roll, yaw, pitch = calculate_head_orientation(
                face_landmarks.landmark, width, height
            )
            tongue_out = is_tongue_out(face_landmarks.landmark, width, height)
            features = {}
            features["LCheekRed"] = left_cheek_redness
            features["RCheekRed"] = right_cheek_redness
            features["LeftEyeAR"] = left_eye_ar
            features["RightEyeAR"] = right_eye_ar
            features["ForeheadRed"] = forehead_redness
            features["LEyeRedness"] = left_eye_redness
            features["REyeRedness"] = right_eye_redness
            features["MouthAR"] = mouth_ar
            features["LBrowDist"] = left_brow_distance
            features["RBrowDist"] = right_brow_distance
            features["Anger"] = emotion_scores["Anger"]
            features["Surprise"] = emotion_scores["Surprise"]
            features["Neutral"] = emotion_scores["Happiness"]
            features["Sadness"] = emotion_scores["Sadness"]
            if mar is not None and curvature is not None:
                smile_metric = curvature / mar
                features["Smile"] = smile_metric
            else:
                features["Smile"] = 0
            features["ForeheadOccluded"] = 0 if forehead_occluded == False else 1
            features["Roll"] = roll
            features["Pitch"] = pitch
            features["Yaw"] = yaw
            features["Tongue"] = 0 if tongue_out == False else 1
        return features
    else:
        return None


os.chdir("/Users/pranavyadlapati/Desktop/MiniProject/new_dataset")
root = "/Users/pranavyadlapati/Desktop/MiniProject/new_dataset"
drunk_path1 = os.path.join(root, "new_drunk_faces")
drunk_path2 = os.path.join(root, "drunk")
sober_path1 = os.path.join(root, "some_more_sober_faces")
sober_path2 = os.path.join(root, "sober")


def generate_data(path):
    results = []
    for filename in os.listdir(path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = cv2.imread(os.path.join(path, filename))
            print(filename)
            if image is None:
                print(f"Error: Could not read image {filename}")
            else:
                result = extract_image(image)
                if result is not None:
                    results.append(result)
        print(filename)
    return results


# drunk_results = generate_data(drunk_path1) + generate_data(drunk_path2)
# sober_results = generate_data(sober_path1) + generate_data(sober_path2)

# drunk_df = pd.DataFrame(drunk_results)
# sober_df = pd.DataFrame(sober_results)
# drunk_df.to_csv("drunk.csv", index=False)
# sober_df.to_csv("sober.csv", index=False)

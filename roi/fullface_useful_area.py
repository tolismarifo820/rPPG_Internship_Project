import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
FACE_OVAL = [p[0] for p in mp_face_mesh.FACEMESH_FACE_OVAL]

EXCLUDE_REGIONS = {
    "left_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "nose": [351, 412, 343, 437, 355, 358, 278, 294, 64, 48, 129, 49, 126, 114, 188, 122, 8, 438, 457, 274, 1, 44, 237, 218],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78],
}

def get_fullface_no_features_mask(image_shape, face_landmarks):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Fill the full face oval using clamped coordinates
    oval_points = np.array([
        [max(0, min(int(face_landmarks.landmark[idx].x * w), w - 1)), 
         max(0, min(int(face_landmarks.landmark[idx].y * h), h - 1))]
        for idx in FACE_OVAL
    ], dtype=np.int32)
    
    oval_hull = cv2.convexHull(oval_points)
    cv2.fillConvexPoly(mask, oval_hull, 255)
    
    # 2. Extract and clamp landmarks for the internal features
    landmarks_px = [
        (max(0, min(int(lm.x * w), w - 1)), max(0, min(int(lm.y * h), h - 1))) 
        for lm in face_landmarks.landmark
    ]
    
    # 3. Explicitly carve out noise features with 0 values
    for region_name, indices in EXCLUDE_REGIONS.items():
        pts = np.array([landmarks_px[i] for i in indices], dtype=np.int32)
        feature_hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [feature_hull], 0)
        
    return mask
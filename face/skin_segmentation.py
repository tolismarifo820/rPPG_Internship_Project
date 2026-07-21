# To call in main:
# from face.skin_segmentation import *

import cv2
import numpy as np
from config import *

# ========================================
# MediaPipe ROI & Masking Core
# ========================================
SKIN_REGIONS = {
    "forehead": [10, 109, 67, 103, 54, 21, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 251, 284, 332, 297, 338],
    "left_upper_cheek": [117, 118, 119, 100, 126, 142, 36, 205, 50, 101],
    "right_upper_cheek": [346, 347, 348, 329, 355, 371, 266, 425, 280, 330],
    "left_temple": [109, 108, 69, 104, 68, 71, 21, 54, 103, 67],
    "right_temple": [338, 337, 299, 333, 298, 301, 251, 284, 332, 297],
    "nose": [351, 412, 343, 437, 355, 358, 278, 294, 64, 48, 129, 49, 126, 114, 188, 122, 8, 438, 457, 274, 1, 44, 237, 218],
}
EXCLUDE_REGIONS = {
    "left_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "left_iris": [474, 475, 476, 477],
    "right_iris": [469, 470, 471, 472],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78],
}

def get_hull_from_indices(landmarks_px, indices):
    pts = np.array([landmarks_px[i] for i in indices], dtype=np.int32)
    return cv2.convexHull(pts)

def extract_mediapipe_roi(frame_bgr, face_mesh):
    h, w, _ = frame_bgr.shape
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    mask = np.zeros((h, w), dtype=np.uint8)
    face_ok, face_box, contours = False, None, []
    
    if results.multi_face_landmarks:
        face_ok = True
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_px = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) for lm in face_landmarks.landmark]
        for region_name, indices in SKIN_REGIONS.items():
            hull = get_hull_from_indices(landmarks_px, indices)
            cv2.fillPoly(mask, [hull], 255)
        for region_name, indices in EXCLUDE_REGIONS.items():
            hull = get_hull_from_indices(landmarks_px, indices)
            cv2.fillPoly(mask, [hull], 0)
        x_coords, y_coords = [p[0] for p in landmarks_px], [p[1] for p in landmarks_px]
        face_box = (min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
    return mask, face_ok, face_box, contours
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
FACE_OVAL = [p[0] for p in mp_face_mesh.FACEMESH_FACE_OVAL]

def get_fullface_mask(image_shape, face_landmarks):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    points = np.array([
        [int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)]
        for idx in FACE_OVAL
    ], dtype=np.int32)
    
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    return mask
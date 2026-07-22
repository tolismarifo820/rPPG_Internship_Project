import cv2
import numpy as np

# MediaPipe indices for left and right cheek patches
LEFT_CHEEK = [47, 100, 119, 101, 118, 117, 116, 36, 50, 123, 205, 206, 207, 187]
RIGHT_CHEEK = [345, 346, 347, 348, 329, 277, 330, 266, 352, 280, 425, 426, 411, 427]

def get_left_cheek_mask(frame_shape, face_landmarks):
    """Generates a mask for the left cheek."""
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    
    points = np.array([
        [int(face_landmarks.landmark[i].x * frame_shape[1]), 
         int(face_landmarks.landmark[i].y * frame_shape[0])] 
        for i in LEFT_CHEEK
    ], dtype=np.int32)
    
    if len(points) > 0:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
    return mask

def get_right_cheek_mask(frame_shape, face_landmarks):
    """Generates a mask for the right cheek."""
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    
    points = np.array([
        [int(face_landmarks.landmark[i].x * frame_shape[1]), 
         int(face_landmarks.landmark[i].y * frame_shape[0])] 
        for i in RIGHT_CHEEK
    ], dtype=np.int32)
    
    if len(points) > 0:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
    return mask
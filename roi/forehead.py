import cv2
import numpy as np

# MediaPipe indices outlining the forehead region (above eyebrows to the hairline)
FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# Note: For a strict forehead, a tighter subset is typically used:
STRICT_FOREHEAD = [10, 109, 67, 103, 54, 21, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 251, 284, 332, 297, 338]

def get_forehead_mask(image_shape, face_landmarks):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert normalized landmarks to pixel coordinates
    points = np.array([
        [int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)]
        for idx in STRICT_FOREHEAD
    ], dtype=np.int32)
    
    # Fill the convex hull of the forehead points
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    return mask
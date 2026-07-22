import cv2
import numpy as np

SKIN_REGIONS = {
    "forehead": [10, 109, 67, 103, 54, 21, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 251, 284, 332, 297, 338],
    "left_upper_cheek": [47, 100, 119, 101, 118, 117, 116, 36, 50, 123, 205, 206, 207, 187],
    "right_upper_cheek": [345, 346, 347, 348, 329, 277, 330, 266, 352, 280, 425, 426, 411, 427],
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

def get_segmented_mask(frame_shape, landmarks):
    """Generates a highly segmented mask including skin regions and excluding features."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate pixel coordinates safely
    landmarks_px = [
        (min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) 
        for lm in landmarks.landmark
    ]
    
    def get_hull(indices):
        pts = np.array([landmarks_px[i] for i in indices], dtype=np.int32)
        return cv2.convexHull(pts)

    # 1. Fill positive skin regions
    for region_name, indices in SKIN_REGIONS.items():
        hull = get_hull(indices)
        cv2.fillPoly(mask, [hull], 255)
        
    # 2. Subtract exclusion regions (eyes, mouth, etc.)
    for region_name, indices in EXCLUDE_REGIONS.items():
        hull = get_hull(indices)
        cv2.fillPoly(mask, [hull], 0)
        
    return mask
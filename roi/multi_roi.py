import cv2
import numpy as np
from roi.roi_manager import extract_mediapipe_roi

def extract_combined_roi(frame, face_mesh, roi_list):
    """Fuses multiple ROIs into a single bounding box and mask."""
    if isinstance(roi_list, str):
        roi_list = [roi_list]
        
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    combined_box = None
    mp_face_ok = False
    mask_contours_all = []

    for roi_name in roi_list:
        mask, ok, box, contours = extract_mediapipe_roi(frame, face_mesh, active_roi_name=roi_name)
        if ok and box is not None:
            mp_face_ok = True
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            mask_contours_all.extend(contours)
            
            if combined_box is None:
                combined_box = list(box)
            else:
                # Dynamically expand the bounding box to encapsulate all ROIs
                x1 = min(combined_box[0], box[0])
                y1 = min(combined_box[1], box[1])
                x2 = max(combined_box[0] + combined_box[2], box[0] + box[2])
                y2 = max(combined_box[1] + combined_box[3], box[1] + box[3])
                combined_box = [x1, y1, x2 - x1, y2 - y1]

    return combined_mask, mp_face_ok, combined_box, mask_contours_all
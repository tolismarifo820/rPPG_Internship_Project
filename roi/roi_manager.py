import cv2
import numpy as np

# Use the dot (.) for explicit relative imports
from .fullface import get_fullface_mask
from .forehead import get_forehead_mask
from .cheeks import get_left_cheek_mask, get_right_cheek_mask
from .segmented import get_segmented_mask

# ---------------------------------------------------------
# HELPER FUNCTIONS (At module level for proper importing)
# ---------------------------------------------------------

def bbox_inside_roi(box, roi):
    """Checks if the face box is completely inside the EVM ROI bounding box."""
    if box is None or roi is None: return False
    bx, by, bw, bh = box
    rx1, ry1, rx2, ry2 = roi
    return (bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2)

def clamp(val, min_val, max_val):
    """Clamps a value between a minimum and maximum boundary."""
    return max(min_val, min(val, max_val))

def extract_candidate_rppg_rois(frame, mp_face_box):
    """Function for ML CSV logger handling candidate ROIs."""
    if mp_face_box is None: return {}
    bx, by, bw, bh = mp_face_box
    return {"Active_ROI": (bx, by, bx+bw, by+bh)}

def handle_roi_keys(key, current_roi_name):
    rois = ["FULL_FACE", "FOREHEAD", "LEFT_CHEEK", "RIGHT_CHEEK", "SEGMENTED"]
    if key == ord('r'):
        current_index = rois.index(current_roi_name) if current_roi_name in rois else 0
        next_index = (current_index + 1) % len(rois)
        new_roi = rois[next_index]
        print(f"ROI switched to: {new_roi}")
        return new_roi
    return current_roi_name

def get_roi_mask(*args, **kwargs):
    """
    Placeholder included to prevent 'ImportError' if main.py still calls 
    this legacy function name. It can be safely ignored.
    """
    pass

# ---------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------------

def extract_mediapipe_roi(frame, face_mesh, active_roi_name="FULL_FACE"):
    """
    Processes the frame with MediaPipe and generates a binary mask and bounding box
    based on the dynamically selected active_roi_name.
    """
    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w = frame.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    mp_face_ok = False
    mp_face_box = None
    mask_contours = []

    if results.multi_face_landmarks:
        mp_face_ok = True
        face_landmarks = results.multi_face_landmarks[0]

        # Route the landmark data to the active ROI module
        if active_roi_name == "FOREHEAD":
            full_mask = get_forehead_mask(frame.shape, face_landmarks)
        elif active_roi_name == "LEFT_CHEEK":
            full_mask = get_left_cheek_mask(frame.shape, face_landmarks)
        elif active_roi_name == "RIGHT_CHEEK":
            full_mask = get_right_cheek_mask(frame.shape, face_landmarks)
        elif active_roi_name == "SEGMENTED":
            full_mask = get_segmented_mask(frame.shape, face_landmarks)
        else:  # Fallback to FULL_FACE
            full_mask = get_fullface_mask(frame.shape, face_landmarks)

        # Extract contours to calculate the bounding box for the UI and EVM crop
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            mask_contours = contours
            
            # For multiple disconnected regions (like cheeks), we need a bounding box 
            # that encompasses all active contours.
            min_x, min_y = w, h
            max_x, max_y = 0, 0
            
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + bw)
                max_y = max(max_y, y + bh)
                
            mp_face_box = (min_x, min_y, max_x - min_x, max_y - min_y)

    return full_mask, mp_face_ok, mp_face_box, mask_contours

def generate_soft_mask(raw_mask, erode_ksize=5, erode_iter=2, blur_ksize=15):
    """Erodes and blurs a binary mask to create a probabilistic float mask."""
    kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
    eroded_mask = cv2.erode(raw_mask, kernel, iterations=erode_iter)
    blurred_mask = cv2.GaussianBlur(eroded_mask, (blur_ksize, blur_ksize), 0)
    
    # Return both the rigid eroded mask (for UI) and the float mask (for EVM/extraction)
    return eroded_mask, blurred_mask.astype(np.float32) / 255.0
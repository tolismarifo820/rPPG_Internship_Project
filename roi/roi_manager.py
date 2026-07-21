# To call in main:
# from roi.roi_manager import *

import cv2
import numpy as np
from config import *

def clamp(val: int, lo: int, hi: int) -> int: return max(lo, min(hi, val))

def face_to_vitals_roi(frame_bgr: np.ndarray, face_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = face_xywh
    x1, x2 = clamp(int(x + 0.20 * w), 0, w_img - 1), clamp(int(x + 0.80 * w), 1, w_img)
    y1, y2 = clamp(int(y + 0.08 * h), 0, h_img - 1), clamp(int(y + 0.35 * h), 1, h_img)
    if x2 <= x1 + 1 or y2 <= y1 + 1: return (0, 0, 0, 0)
    return (x1, y1, x2, y2)

def safe_roi_from_face(x1, y1, x2, y2, img_w, img_h):
    x1, y1 = clamp(int(x1), 0, img_w - 1), clamp(int(y1), 0, img_h - 1)
    x2, y2 = clamp(int(x2), 1, img_w), clamp(int(y2), 1, img_h)
    if x2 <= x1 + 2 or y2 <= y1 + 2: return None
    return (x1, y1, x2, y2)

def extract_candidate_rppg_rois(frame_bgr: np.ndarray, face_xywh: tuple[int, int, int, int] | None):
    if face_xywh is None: return {}
    img_h, img_w = frame_bgr.shape[:2]
    x, y, w, h = face_xywh
    rois = {}

    current = face_to_vitals_roi(frame_bgr, face_xywh)
    if current != (0, 0, 0, 0): rois["forehead_current"] = current

    roi = safe_roi_from_face(x + 0.12 * w, y + 0.07 * h, x + 0.88 * w, y + 0.32 * h, img_w, img_h)
    if roi is not None: rois["forehead_wide"] = roi
    roi = safe_roi_from_face(x + 0.14 * w, y + 0.42 * h, x + 0.40 * w, y + 0.68 * h, img_w, img_h)
    if roi is not None: rois["left_cheek"] = roi
    roi = safe_roi_from_face(x + 0.60 * w, y + 0.42 * h, x + 0.86 * w, y + 0.68 * h, img_w, img_h)
    if roi is not None: rois["right_cheek"] = roi
    roi = safe_roi_from_face(x + 0.12 * w, y + 0.42 * h, x + 0.88 * w, y + 0.68 * h, img_w, img_h)
    if roi is not None: rois["both_cheeks_combined"] = roi
    roi = safe_roi_from_face(x + 0.42 * w, y + 0.30 * h, x + 0.58 * w, y + 0.52 * h, img_w, img_h)
    if roi is not None: rois["nose_bridge"] = roi
    roi = safe_roi_from_face(x + 0.12 * w, y + 0.12 * h, x + 0.88 * w, y + 0.78 * h, img_w, img_h)
    if roi is not None: rois["full_face_proxy"] = roi
    return rois

def bbox_inside_roi(face_xywh: tuple[int, int, int, int] | None, roi_bbox: tuple[int, int, int, int]) -> bool:
    if face_xywh is None: return False
    fx, fy, fw, fh = face_xywh
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    x1, y1, x2, y2 = roi_bbox
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)
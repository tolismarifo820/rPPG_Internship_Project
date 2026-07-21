# To call in main:
# from signals.filtering import *

import cv2
import numpy as np
from config import *

def bgr_to_yiq(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(img)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B
    return Y, I, Q

def yiq_to_bgr(Y, I, Q):
    R = Y + 0.956 * I + 0.621 * Q
    G = Y - 0.272 * I - 0.647 * Q
    B = Y - 1.106 * I + 1.703 * Q
    img = cv2.merge([B, G, R])
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

def build_laplacian_pyr(img, levels_count):
    pyr = []
    current = img.astype(np.float32)
    for _ in range(levels_count):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        lap = current - up
        pyr.append(lap)
        current = down
    pyr.append(current)
    return pyr

def collapse_laplacian_pyr(pyr):
    current = pyr[-1]
    for level in reversed(pyr[:-1]):
        up = cv2.pyrUp(current, dstsize=(level.shape[1], level.shape[0]))
        current = up + level
    return current
# To call in main:
# from config import *

from __future__ import annotations

import os
import csv
import time
import datetime as dt
import json
import traceback
from collections import deque

import cv2
import numpy as np

# ========================================
# MEDIAPIPE EXPLICIT IMPORT FIX
# ========================================
try:
    import mediapipe as mp
    try:
        # Standard dynamic attribute access
        mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        # Fallback absolute import if the dynamic namespace fails to load
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    # This will print the ACTUAL error if it fails, instead of lying about the installation
    raise SystemExit(f"MediaPipe is installed, but crashed during import. Exact error: {type(e).__name__}: {e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

import threading
from queue import Queue, Empty

# ========================================
# Configuration
# ========================================
realWidth, realHeight = 320, 240
videoWidth, videoHeight = 160, 120
videoChannels = 3
fps = 20

# PC camera configuration
PC_CAMERA_INDEX = 0
USE_PICAMERA2 = False

# EVM & Signal Extraction limits
levels = 4
alpha = 80.0
minFrequency = 0.8          
maxFrequency = 3          
chromAttenuation = 0.2
bufferSize = 75             
bpmBufferSize = 5        
bpmCalcEvery = 5           

SPO2_A = 110
SPO2_B = 25

hr_low, hr_high = 0.7, 3.0  
rr_low, rr_high = 0.1, 0.5

FACE_STABLE_SECONDS_REQUIRED = 10.0
FADE_IN_SECONDS = 1.2

stop_event = threading.Event()

WINDOW_NAME = "RPi Medical Bento Dashboard"
SCREEN_W = 1280
SCREEN_H = 720
CANVAS_W, CANVAS_H = 1280, 720
PAD = 10

DARK_THEME = {
    "BG_COLOR": (18, 18, 18), "CARD_COLOR": (70, 70, 70), "BORDER_COLOR": (12, 60, 90),
    "TITLE_COLOR": (235, 235, 235), "SUBTLE_TEXT": (160, 160, 160), "ACCENT_PULSE": (255, 80, 80),
    "ACCENT_BREATH": (50, 170, 255), "ACCENT_OXY": (75, 215, 50), "ACCENT_EMOTION": (230, 170, 210),
    "ACCENT_WARNING": (0, 180, 255), "ACCENT_EVM": (0, 255, 0), "ACCENT_OK": (80, 220, 120),
    "ACCENT_INFO": (255, 210, 80), "ACCENT_ERROR": (90, 90, 255),
}

LIGHT_THEME = {
    "BG_COLOR": (255, 255, 255), "CARD_COLOR": (245, 247, 250), "BORDER_COLOR": (200, 210, 220),
    "TITLE_COLOR": (40, 40, 40), "SUBTLE_TEXT": (110, 110, 110), "ACCENT_PULSE": (220, 60, 60),
    "ACCENT_BREATH": (60, 140, 255), "ACCENT_OXY": (40, 180, 90), "ACCENT_EMOTION": (200, 120, 200),
    "ACCENT_WARNING": (0, 140, 255), "ACCENT_EVM": (0, 200, 0), "ACCENT_OK": (60, 180, 90),
    "ACCENT_INFO": (255, 180, 60), "ACCENT_ERROR": (80, 80, 220),
}

ACTIVE_THEME_NAME = "light"
ACQUISITION_ACTIVE_DEFAULT = False
SQI_WINDOW = 90
SQI_GOOD_THRESHOLD = 70.0
SQI_FAIR_THRESHOLD = 45.0

LOG_ROOT = "logs"

PROTOCOL_POSITIONING = "POSITIONING"
PROTOCOL_LOCKING = "LOCKING"
PROTOCOL_ACQUISITION = "ACQUISITION"
PROTOCOL_COMPLETE = "COMPLETE"
ACQUISITION_DURATION_SECONDS = 30.0

FACE_LOSS_RESET_SECONDS = 2.0
POOR_SQI_RESET_SECONDS = 3.0
POOR_SQI_THRESHOLD = 25.0
VALID_MEAN_SQI_THRESHOLD = 45.0

def apply_theme(theme_name: str):
    global ACTIVE_THEME_NAME
    ACTIVE_THEME_NAME = theme_name
    theme = DARK_THEME if theme_name == "dark" else LIGHT_THEME
    globals().update(theme)

apply_theme(ACTIVE_THEME_NAME)

# UI Layout Coordinates
LEFT_W, CENTER_W, RIGHT_W = 360, 560, 320
TOP_H, MID_H, BOT_H = 210, 210, 270

BOX_PULSE = (PAD, PAD, LEFT_W, TOP_H)
BOX_WAVE = (PAD, PAD * 2 + TOP_H, LEFT_W, MID_H)
BOX_EMOTION = (PAD, PAD * 3 + TOP_H + MID_H, LEFT_W, BOT_H)
BOX_STATUS = (PAD * 2 + LEFT_W, PAD, CENTER_W, TOP_H + MID_H)
BOTTOM_X = PAD * 2 + LEFT_W
BOTTOM_W = CENTER_W + PAD + RIGHT_W
BOX_BR = (BOTTOM_X, PAD * 3 + TOP_H + MID_H, BOTTOM_W // 2 - PAD // 2, BOT_H)
BOX_SPO2 = (BOTTOM_X + BOTTOM_W // 2 + PAD // 2, PAD * 3 + TOP_H + MID_H, BOTTOM_W // 2 - PAD // 2, BOT_H)
BOX_LOGO = (PAD * 3 + LEFT_W + CENTER_W, PAD, RIGHT_W, TOP_H)
BOX_CAMERA = (PAD * 3 + LEFT_W + CENTER_W, PAD * 2 + TOP_H, RIGHT_W, MID_H)
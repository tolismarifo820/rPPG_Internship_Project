import os, time, traceback
import datetime as dt
import cv2
import numpy as np
import polars as pl
from collections import deque
import mediapipe as mp
from scipy.signal import butter, filtfilt
from scipy import fft as sp_fft

# ==============================================================================
# DEVICE & SAVING TOGGLES
# ==============================================================================
RUN_ON_LAPTOP = 1
SAVE_METADATA = 1
SAVE_CAMERA_ACQUISITION = 1

# Raspberry Pi runtime tuning. These settings do not change the algorithm;
# they reduce thread contention between OpenCV, SciPy FFT, and MediaPipe.
if not RUN_ON_LAPTOP:
    CV2_THREADS = min(2, os.cpu_count() or 1)
    FFT_WORKERS = min(2, os.cpu_count() or 1)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(CV2_THREADS)
else:
    FFT_WORKERS = 1

if RUN_ON_LAPTOP:
    realWidth, realHeight = 1280, 720
    PC_CAMERA_INDEX = 0
    OUTPUT_DIR = r"C:\Users\tolis\Desktop\Praktiki\rPPG_project\logs_tif90"
else:
    realWidth, realHeight = 640, 480
    OUTPUT_DIR = "/home/mpl/scripts/tif90"
    try:
        from picamera2 import Picamera2
    except ImportError:
        pass 

    try:
        mp_face_mesh_module = mp.solutions.face_mesh
    except AttributeError:
        try:
            import mediapipe.python.solutions.face_mesh as mp_face_mesh_module
        except ImportError:
            mp_face_mesh_module = mp.python.solutions.face_mesh

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CONFIGURATION VARIABLES
# ==============================================================================
fps = 15.0
bufferSize = int(fps * 10)  
bpmBufferSize = 10
bpmCalcEvery = int(fps * 1) 
FACE_STABLE_SECONDS_REQUIRED = 5.0  

ACQUISITION_DURATION_SEC = 20.0
TOTAL_RECORD_FRAMES = int(fps * ACQUISITION_DURATION_SEC)

hr_low, hr_high = 0.7, 3.0   
rr_low, rr_high = 0.15, 0.5  
SPO2_A, SPO2_B = 120.0, 25.0

levels = 3                  
alpha_bgr = 50.0            
alpha_yiq = 50.0            
minFrequency = 0.7          
maxFrequency = 3.0          

WINDOW_NAME = "Vitals Dashboard"
CANVAS_W, CANVAS_H = 880, 520
SCREEN_W, SCREEN_H = 880, 520

BOX_PULSE = (30, 30, 240, 130)
BOX_BR = (30, 180, 240, 130)
BOX_SPO2 = (30, 330, 240, 130)
BOX_CAMERA = (300, 30, 540, 460)

BG_COLOR = (18, 18, 22)
CARD_COLOR = (30, 30, 38)
BORDER_COLOR = (55, 55, 68)
TITLE_COLOR = (140, 140, 160)
SUBTLE_TEXT = (100, 100, 120)
ACCENT_PULSE = (75, 75, 245)
ACCENT_BREATH = (245, 160, 60)
ACCENT_OXY = (60, 220, 120)
ACCENT_EVM = (200, 200, 0)
RECORDING_COLOR = (0, 0, 255)

# ==============================================================================
# ROI MASKS
# ==============================================================================
SKIN_REGIONS = {
    "forehead": [10, 109, 67, 103, 54, 21, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 251, 284, 332, 297, 338]
}

# Only these landmark indices are ever used by the configured ROI(s).
ROI_LANDMARK_INDICES = tuple(sorted({i for indices in SKIN_REGIONS.values() for i in indices}))


def get_segmented_mask(frame_shape, landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # The original code converted all 468 landmarks to pixels even though
    # only the configured skin-region indices are consumed.
    point_map = {}
    for idx in ROI_LANDMARK_INDICES:
        lm = landmarks.landmark[idx]
        point_map[idx] = (
            min(int(lm.x * w), w - 1),
            min(int(lm.y * h), h - 1),
        )

    for indices in SKIN_REGIONS.values():
        region_points = np.asarray([point_map[i] for i in indices], dtype=np.int32)
        cv2.fillPoly(mask, [cv2.convexHull(region_points)], 255)

    return mask

def bbox_inside_roi(box, roi):
    if box is None or roi is None: return False
    bx, by, bw, bh = box; rx1, ry1, rx2, ry2 = roi
    return (bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2)

def scale_mask_contours(mask_contours, scale_x, scale_y, offset_x, offset_y):
    if not mask_contours:
        return []
    scaled = []
    for cnt in mask_contours:
        out = np.empty_like(cnt)
        out[:, 0, 0] = (cnt[:, 0, 0] * scale_x + offset_x).astype(np.int32)
        out[:, 0, 1] = (cnt[:, 0, 1] * scale_y + offset_y).astype(np.int32)
        scaled.append(out)
    return scaled

def clamp(val, min_val, max_val): return max(min_val, min(val, max_val))

def extract_mediapipe_roi(frame, face_mesh, scale=0.5):
    h, w = frame.shape[:2]
    small_w, small_h = int(w * scale), int(h * scale)
    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    mp_face_ok, mp_face_box, mask_contours = False, None, []

    if results.multi_face_landmarks:
        mp_face_ok = True
        face_landmarks = results.multi_face_landmarks[0]
        small_mask = get_segmented_mask((small_h, small_w), face_landmarks)
        full_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask_contours = contours
            # Same bounding rectangle as the original contour min/max loop,
            # but computed inside OpenCV's optimized C implementation.
            mp_face_box = cv2.boundingRect(full_mask)
            
    return full_mask, mp_face_ok, mp_face_box, mask_contours

# ==============================================================================
# COLOR SPACES & SIGNAL PROCESSING
# ==============================================================================
# Same linear transforms as the original channel-wise equations, expressed
# as OpenCV matrix transforms so the work runs in optimized native code.
BGR_TO_YIQ = np.array([
    [0.114, 0.587, 0.299],
    [-0.322, -0.274, 0.596],
    [0.312, -0.523, 0.211],
], dtype=np.float32)

YIQ_TO_BGR = np.array([
    [1.0, -1.106, 1.703],
    [1.0, -0.272, -0.647],
    [1.0, 0.956, 0.621],
], dtype=np.float32)

_FILTER_CACHE = {}

def bgr_to_yiq(img_bgr):
    img = img_bgr.astype(np.float32, copy=False)
    img *= (1.0 / 255.0)
    yiq = cv2.transform(img, BGR_TO_YIQ)
    return yiq[:, :, 0], yiq[:, :, 1], yiq[:, :, 2]

def yiq_to_bgr(Y, I, Q):
    yiq = cv2.merge((Y, I, Q))
    img = cv2.transform(yiq, YIQ_TO_BGR)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    if len(data) < 15:
        return data

    key = (float(lowcut), float(highcut), float(fs), int(order))
    coeffs = _FILTER_CACHE.get(key)
    if coeffs is None:
        coeffs = butter(
            order,
            [lowcut / (0.5 * fs), min(0.99, highcut / (0.5 * fs))],
            btype='band'
        )
        _FILTER_CACHE[key] = coeffs

    b, a = coeffs
    try:
        return filtfilt(
            b, a, data,
            padlen=min(len(data) - 1, 3 * max(len(b), len(a)))
        )
    except ValueError:
        return data

def extract_pos(r, g, b, fps_val):
    N, l = len(r), int(2 * fps_val)
    if N < l: return np.zeros(N)
    eps = 1e-6
    rw, gw, bw = np.lib.stride_tricks.sliding_window_view(r, l), np.lib.stride_tricks.sliding_window_view(g, l), np.lib.stride_tricks.sliding_window_view(b, l)
    rn, gn, bn = rw/(np.mean(rw, axis=1, keepdims=True)+eps), gw/(np.mean(gw, axis=1, keepdims=True)+eps), bw/(np.mean(bw, axis=1, keepdims=True)+eps)
    S1, S2 = gn - bn, gn + bn - 2 * rn
    h = S1 + (np.std(S1, axis=1, keepdims=True) / (np.std(S2, axis=1, keepdims=True) + eps)) * S2
    h_zm = h - np.mean(h, axis=1, keepdims=True)
    H = np.zeros(N)
    for i in range(l): H[i : i + h_zm.shape[0]] += h_zm[:, i]
    return H

_PEAK_N = 1024
_PEAK_FREQS = np.fft.rfftfreq(_PEAK_N, d=1.0 / fps)
_HR_PEAK_MASK = (_PEAK_FREQS >= hr_low) & (_PEAK_FREQS <= hr_high)
_RR_PEAK_MASK = (_PEAK_FREQS >= rr_low) & (_PEAK_FREQS <= rr_high)


def estimate_peak_bpm(signal, fps_val, min_hz, max_hz):
    # Only positive frequencies are inspected by the original code, so rFFT
    # returns the same candidate magnitudes while doing roughly half the work.
    mag = np.abs(sp_fft.rfft(
        signal - float(np.mean(signal)),
        n=_PEAK_N,
        workers=FFT_WORKERS
    ))

    if min_hz == hr_low and max_hz == hr_high:
        peak_mask = _HR_PEAK_MASK
    elif min_hz == rr_low and max_hz == rr_high:
        peak_mask = _RR_PEAK_MASK
    else:
        freqs = _PEAK_FREQS
        peak_mask = (freqs >= min_hz) & (freqs <= max_hz)

    return float(_PEAK_FREQS[int(np.argmax(mag * peak_mask))] * 60.0) if np.any(peak_mask) else 0.0

def robust_mean(data):
    if not data: return None
    return float(np.median(list(data))) if len(data) >= 3 else float(np.mean(list(data)))

def safe_round(value, decimals=2):
    return round(float(value), decimals) if value is not None and not np.isnan(value) else None

# ==============================================================================
# UI RENDERING & I/O LOGGING
# ==============================================================================
def open_acquisition_video_writer(sess_dir: str, frame, fps_value: float):
    try:
        h, w = frame.shape[:2]
        video_path = os.path.join(sess_dir, "acquisition.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, float(fps_value), (w, h))
        
        if not writer.isOpened():
            video_path = os.path.join(sess_dir, "acquisition.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(video_path, fourcc, float(fps_value), (w, h))
            
        if not writer.isOpened(): 
            return None
        return writer
    except Exception: 
        return None

def display_metric_value(face_detected_now, vitals_enabled, value):
    if not face_detected_now: return None
    if not vitals_enabled: return "Calculating..."
    return value

def draw_card(canvas, target_box, title, value, unit, color, show_pulse_icon=False, current_bpm=None):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)
    cv2.putText(canvas, title, (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, TITLE_COLOR, 2, cv2.LINE_AA)

    if isinstance(value, str): val_str, font_scale, thickness = value, 1.0, 2
    elif value is not None and value > 0: val_str, font_scale, thickness = f"{value:.0f}", 2.4, 3
    else: val_str, font_scale, thickness = "--", 2.4, 3

    (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    cv2.putText(canvas, val_str, (x + max(24, (w - tw) // 2), y + h // 2 + th // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)
    if unit: cv2.putText(canvas, unit, (x + 24, y + h - 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, SUBTLE_TEXT, 1, cv2.LINE_AA)
    
    if show_pulse_icon and isinstance(current_bpm, (int, float, np.floating)) and current_bpm > 30:
        pulse_scale = 0.75 + 0.25 * np.sin((time.time() * (float(current_bpm) / 60.0) * 2 * np.pi) % (2 * np.pi))
        pts = (np.array([[0, -8], [8, -15], [15, -8], [0, 12], [-15, -8], [-8, -15]], np.int32) * pulse_scale * 1.4).astype(np.int32)
        cv2.fillPoly(canvas, [pts + [x + w - 52, y + 56]], color, lineType=cv2.LINE_AA)

def draw_camera_card(canvas, target_box, frame, evm_roi_bbox=None, mask_contours=None, scaled_mask_contours=None, is_recording=False, remaining_sec=20, participant_id="", evm_mode="BGR"):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cam_h, cam_w = h - 20, w - 20
    actual_h, actual_w = frame.shape[:2]
    canvas[y + 10:y + 10 + cam_h, x + 10:x + 10 + cam_w] = cv2.resize(frame, (cam_w, cam_h))
    scale_x, scale_y = cam_w / actual_w, cam_h / actual_h

    if evm_roi_bbox is not None:
        ex1, ey1, ex2, ey2 = evm_roi_bbox
        cv2.rectangle(canvas, (int(ex1 * scale_x) + x + 10, int(ey1 * scale_y) + y + 10), (int(ex2 * scale_x) + x + 10, int(ey2 * scale_y) + y + 10), ACCENT_EVM, 2, cv2.LINE_AA)
    
    if scaled_mask_contours is not None:
        cv2.polylines(
            canvas, scaled_mask_contours, isClosed=True,
            color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )

    cv2.rectangle(canvas, (x, y), (x + w, y + h), RECORDING_COLOR if is_recording else BORDER_COLOR, 2)
    
    overlay_x = x + 18
    overlay_y = y + h - 75
    
    cv2.putText(canvas, f"Participant: {participant_id}", (overlay_x, overlay_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    rec_status_text = f"Recording: {'ACTIVE (' + str(int(remaining_sec)) + 's remaining)' if is_recording else 'INACTIVE'}"
    cv2.putText(canvas, rec_status_text, (overlay_x, overlay_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, RECORDING_COLOR if is_recording else (180, 180, 180), 1, cv2.LINE_AA)
    toggles_text = f"Toggles: [SPACE] Rec | [N] New Part | [E] EVM ({evm_mode})"
    cv2.putText(canvas, toggles_text, (overlay_x, overlay_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

def flush_metadata_to_parquet(records, sess_dir):
    if records:
        df = pl.DataFrame(records)
        df.to_parquet(os.path.join(sess_dir, "metadata.parquet"), index=False)

# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================
def main():
    cap, picam2 = None, None
    try:
        if RUN_ON_LAPTOP:
            cap = cv2.VideoCapture(PC_CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
            cap.set(cv2.CAP_PROP_FPS, fps)
            time.sleep(0.5)
            mp_face_mesh = mp.solutions.face_mesh
        else:
            picam2 = Picamera2()
            picam2.configure(picam2.create_video_configuration(main={"size": (realWidth, realHeight), "format": "BGR888"}))
            picam2.start()
            time.sleep(2.0)
            mp_face_mesh = mp_face_mesh_module

        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        red_buffer, green_buffer, blue_buffer = [np.zeros((bufferSize,), dtype=np.float32) for _ in range(3)]
        bpmBuffer = np.full((bpmBufferSize,), np.nan, dtype=np.float32)
        bpm_all, rr_history, spo2_history = [], deque(maxlen=10), deque(maxlen=10)
        
        buffers_initialized, video_pyr_initialized = False, False
        bufferIndex, bpmBufferIndex = 0, 0
        current_hr, current_rr, current_spo2 = None, None, None
        smoothed_bbox = None
        
        face_detected_now, vitals_enabled = False, False
        stable_face_start_time = None
        
        videoPyramid = None
        freqs = np.fft.fftfreq(bufferSize, d=1.0 / fps)
        mask = (freqs >= minFrequency) & (freqs <= maxFrequency)
        fft_mask = mask.astype(np.float32)[:, None, None, None]
        active_evm_mode = "BGR"

        cached_full_mask, cached_mp_face_ok, cached_mp_face_box, cached_mask_contours = None, False, None, []
        cached_scaled_mask_contours = []
        frame_counter = 0

        # Reuse the dashboard canvas instead of allocating ~1.4 MB every frame.
        canvas = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # Reuse POS circular-buffer reorder storage.
        r_rolled = np.empty(bufferSize, dtype=np.float32)
        g_rolled = np.empty(bufferSize, dtype=np.float32)
        b_rolled = np.empty(bufferSize, dtype=np.float32)

        # Initialize Participant ID Sequence (P001, P002, etc.)
        participant_counter = 1
        existing_sessions = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("P") and len(d) == 4 and d[1:].isdigit()]
        for d in existing_sessions:
            try:
                num = int(d[1:])
                if num >= participant_counter: participant_counter = num + 1
            except ValueError:
                pass
        
        current_participant_id = f"P{participant_counter:03d}"

        # Recording State
        is_recording = False
        record_frames = 0
        video_writer = None
        metadata_records = []
        sess_dir = ""

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL if RUN_ON_LAPTOP else cv2.WINDOW_AUTOSIZE)
        if RUN_ON_LAPTOP: cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)

        while True:
            loop_start = time.time()
            frame_counter += 1
            
            if RUN_ON_LAPTOP:
                ret, frame = cap.read()
                if not ret or frame is None: time.sleep(0.05); continue
            else:
                try: frame = picam2.capture_array()
                except Exception: time.sleep(0.05); continue

            frame = cv2.flip(frame, 1)
            # Only copy the frame when the acquisition writer will use it.
            raw_frame_for_video = (
                frame.copy()
                if (is_recording and SAVE_CAMERA_ACQUISITION and video_writer is not None)
                else None
            )

            current_h, current_w = frame.shape[:2]
            evm_roi_bbox = (int(current_w * 0.05), int(current_h * 0.05), current_w - int(current_w * 0.05), current_h - int(current_h * 0.05))

            if frame_counter % 2 != 0 or not cached_mp_face_ok:
                full_mask, mp_face_ok, mp_face_box, mask_contours = extract_mediapipe_roi(frame, face_mesh, scale=0.5)
                cached_full_mask, cached_mp_face_ok, cached_mp_face_box, cached_mask_contours = full_mask, mp_face_ok, mp_face_box, mask_contours

                cached_scaled_mask_contours = scale_mask_contours(
                    mask_contours,
                    (BOX_CAMERA[2] - 20) / current_w,
                    (BOX_CAMERA[3] - 20) / current_h,
                    BOX_CAMERA[0] + 10,
                    BOX_CAMERA[1] + 10
                )
            else:
                full_mask, mp_face_ok, mp_face_box, mask_contours = cached_full_mask, cached_mp_face_ok, cached_mp_face_box, cached_mask_contours

            face_detected_now = mp_face_ok and bbox_inside_roi(mp_face_box, evm_roi_bbox)

            if face_detected_now:
                if stable_face_start_time is None: stable_face_start_time = time.time()
                vitals_enabled = (time.time() - stable_face_start_time) >= FACE_STABLE_SECONDS_REQUIRED
            else:
                stable_face_start_time, vitals_enabled = None, False

            vitals_roi, vitals_mask = None, None
            if mp_face_box is not None and face_detected_now:
                raw_x, raw_y, raw_w, raw_h = mp_face_box
                if smoothed_bbox is None or not face_detected_now: smoothed_bbox = [raw_x, raw_y, raw_w, raw_h]
                else:
                    alpha_s = 0.15 
                    smoothed_bbox = [(1.0 - alpha_s)*s + alpha_s*r for s, r in zip(smoothed_bbox, [raw_x, raw_y, raw_w, raw_h])]

                vx1, vy1, vw, vh = [int(v) for v in smoothed_bbox]
                vx2, vy2 = clamp(vx1 + vw, 1, current_w), clamp(vy1 + vh, 1, current_h)
                vx1, vy1 = clamp(vx1, 0, current_w - 1), clamp(vy1, 0, current_h - 1)
                
                if vx2 > vx1 and vy2 > vy1:
                    vitals_roi, vitals_mask = frame[vy1:vy2, vx1:vx2, :], full_mask[vy1:vy2, vx1:vx2]

            if vitals_roi is not None and vitals_mask is not None:
                # 1. Read clean pixel data directly BEFORE applying EVM
                b_val, g_val, r_val = [float(v) for v in cv2.mean(vitals_roi, mask=vitals_mask)[:3]]
                if r_val == 0.0 and g_val == 0.0 and b_val == 0.0 and buffers_initialized:
                    r_val, g_val, b_val = float(red_buffer[bufferIndex-1]), float(green_buffer[bufferIndex-1]), float(blue_buffer[bufferIndex-1])

                if not buffers_initialized:
                    red_buffer[:], green_buffer[:], blue_buffer[:] = r_val, g_val, b_val
                    buffers_initialized = True
                else:
                    red_buffer[bufferIndex], green_buffer[bufferIndex], blue_buffer[bufferIndex] = r_val, g_val, b_val

                # 2. EVM magnification for display output only
                processed_roi = (
                    cv2.merge(bgr_to_yiq(vitals_roi))
                    if active_evm_mode == "YIQ"
                    else vitals_roi.astype(np.float32, copy=False)
                )

                # Only target_level and target_level+1 are consumed by the
                # original code, so do not keep unused intermediate levels.
                target_level = levels - 1
                low = processed_roi
                for _ in range(target_level):
                    low = cv2.pyrDown(low)
                lower = cv2.pyrDown(low)
                laplacian = cv2.subtract(
                    low,
                    cv2.pyrUp(lower, dstsize=(low.shape[1], low.shape[0]))
                )

                if not video_pyr_initialized:
                    if (
                        videoPyramid is None
                        or videoPyramid.shape[1] != laplacian.shape[0]
                        or videoPyramid.shape[2] != laplacian.shape[1]
                    ):
                        videoPyramid = np.zeros(
                            (bufferSize, laplacian.shape[0], laplacian.shape[1], 3),
                            dtype=np.float32
                        )
                    else:
                        videoPyramid.fill(0.0)
                    video_pyr_initialized = True
                elif laplacian.shape[:2] != videoPyramid.shape[1:3]:
                    laplacian = cv2.resize(
                        laplacian,
                        (videoPyramid.shape[2], videoPyramid.shape[1])
                    )

                videoPyramid[bufferIndex] = laplacian

                # The original code rolls the circular buffer before filtering
                # and reads the last sample. FFT filtering is shift-invariant,
                # so the equivalent physical sample is bufferIndex. This removes
                # a full-buffer memory copy every frame.
                fft_buffer = sp_fft.fft(
                    videoPyramid, axis=0, workers=FFT_WORKERS
                )
                fft_buffer *= fft_mask
                filtered_laplacian = (
                    sp_fft.ifft(
                        fft_buffer, axis=0, workers=FFT_WORKERS
                    )[bufferIndex].real
                    * (alpha_yiq if active_evm_mode == "YIQ" else alpha_bgr)
                )

                amplified_signal = filtered_laplacian
                for _ in range(target_level): 
                    amplified_signal = cv2.pyrUp(amplified_signal)
                
                amplified_resized = cv2.resize(
                    amplified_signal,
                    (vitals_roi.shape[1], vitals_roi.shape[0])
                )
                masked_pulse = np.empty_like(amplified_resized)
                np.multiply(
                    amplified_resized,
                    (vitals_mask > 0)[..., None],
                    out=masked_pulse
                )
                np.clip(masked_pulse, -15.0, 15.0, out=masked_pulse)

                if active_evm_mode == "YIQ":
                    out_Y, out_I, out_Q = cv2.split(processed_roi + masked_pulse)
                    frame[vy1:vy2, vx1:vx2, :] = yiq_to_bgr(out_Y, out_I, out_Q)
                else:
                    frame[vy1:vy2, vx1:vx2, :] = np.clip(processed_roi + masked_pulse, 0, 255).astype(np.uint8)

            # 3. Calculate vitals from clean historical buffers.
            # In the original program these calculations only affect displayed
            # vitals once per second, so running them on intermediate frames was
            # unused work.
            if (
                buffers_initialized
                and face_detected_now
                and vitals_enabled
                and bufferIndex % bpmCalcEvery == 0
            ):
                split = bufferIndex + 1
                tail = bufferSize - split

                if tail > 0:
                    r_rolled[:tail] = red_buffer[split:]
                    r_rolled[tail:] = red_buffer[:split]
                    g_rolled[:tail] = green_buffer[split:]
                    g_rolled[tail:] = green_buffer[:split]
                    b_rolled[:tail] = blue_buffer[split:]
                    b_rolled[tail:] = blue_buffer[:split]
                else:
                    r_rolled[:] = red_buffer
                    g_rolled[:] = green_buffer
                    b_rolled[:] = blue_buffer

                raw_extracted_signal = extract_pos(
                    r_rolled, g_rolled, b_rolled, int(fps)
                )
                active_signal = bandpass_filter(
                    raw_extracted_signal, hr_low, hr_high, fps, order=2
                )

                bpm = estimate_peak_bpm(
                    active_signal, fps, hr_low, hr_high
                )
                if len(bpm_all) > 0:
                    bpm = float(np.clip(
                        bpm,
                        bpm_all[-1] - 3.0 * (bpmCalcEvery / fps),
                        bpm_all[-1] + 3.0 * (bpmCalcEvery / fps)
                    ))

                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
                bpm_all.append(bpm)
                current_hr = np.nanmean(bpmBuffer)

                rr_raw = estimate_peak_bpm(
                    bandpass_filter(r_rolled, rr_low, rr_high, fps, order=2),
                    fps, rr_low, rr_high
                )
                if 6 <= rr_raw <= 40:
                    rr_history.append(rr_raw)
                current_rr = robust_mean(rr_history)

                # Cardiac-filtered SpO2 calculation
                r_hr_band = bandpass_filter(
                    r_rolled, hr_low, hr_high, fps, order=2
                )
                b_hr_band = bandpass_filter(
                    b_rolled, hr_low, hr_high, fps, order=2
                )

                ac_r = float(np.std(r_hr_band))
                dc_r = float(np.mean(r_rolled))
                ac_b = float(np.std(b_hr_band))
                dc_b = float(np.mean(b_rolled))

                if dc_r > 1e-3 and dc_b > 1e-3 and ac_b > 1e-6:
                    ratio_of_ratios = (ac_r / dc_r) / (ac_b / dc_b)
                    spo2_raw = min(
                        float(SPO2_A - (SPO2_B * ratio_of_ratios)), 100.0
                    )

                    if 85.0 <= spo2_raw <= 100.0:
                        spo2_history.append(spo2_raw)

                spo2_smoothed = robust_mean(spo2_history)
                current_spo2 = (
                    spo2_smoothed if spo2_smoothed is not None else None
                )

            if not face_detected_now:
                current_hr, current_rr, current_spo2 = None, None, None
                buffers_initialized, video_pyr_initialized = False, False
                if videoPyramid is not None:
                    videoPyramid.fill(0.0)
                rr_history.clear(); spo2_history.clear(); bpm_all.clear()
                bpmBuffer[:] = np.nan

            remaining_sec = max(0.0, ACQUISITION_DURATION_SEC - (record_frames / fps))

            if is_recording:
                if SAVE_CAMERA_ACQUISITION and video_writer is not None and raw_frame_for_video is not None:
                    video_writer.write(raw_frame_for_video)
                if SAVE_METADATA:
                    metadata_records.append({
                        "timestamp": dt.datetime.now().isoformat(),
                        "frame_idx": record_frames,
                        "hr_bpm": safe_round(current_hr, 2),
                        "rr_brpm": safe_round(current_rr, 2),
                        "spo2%": safe_round(current_spo2, 2)
                    })
                
                record_frames += 1
                if record_frames >= TOTAL_RECORD_FRAMES:
                    is_recording = False
                    if SAVE_METADATA:
                        flush_metadata_to_parquet(metadata_records, sess_dir)
                        metadata_records.clear()
                    if video_writer: 
                        video_writer.release()
                        video_writer = None

            canvas[:] = BG_COLOR
            draw_card(canvas, BOX_PULSE, "HEART RATE", display_metric_value(face_detected_now, vitals_enabled, current_hr), "BPM", ACCENT_PULSE, show_pulse_icon=True, current_bpm=current_hr if vitals_enabled else None)
            draw_card(canvas, BOX_BR, "RESPIRATION", display_metric_value(face_detected_now, vitals_enabled, current_rr), "BR/MIN", ACCENT_BREATH)
            draw_card(canvas, BOX_SPO2, "OXYGEN", display_metric_value(face_detected_now, vitals_enabled, current_spo2), "SpO2 %", ACCENT_OXY)
            draw_camera_card(
                canvas, BOX_CAMERA, frame,
                evm_roi_bbox=evm_roi_bbox,
                mask_contours=mask_contours,
                scaled_mask_contours=cached_scaled_mask_contours,
                is_recording=is_recording,
                remaining_sec=remaining_sec,
                participant_id=current_participant_id,
                evm_mode=active_evm_mode
            )

            cv2.imshow(WINDOW_NAME, canvas)
            bufferIndex = (bufferIndex + 1) % bufferSize
            
            wait_ms = max(1, int(max(0.0, (1.0 / fps) - (time.time() - loop_start)) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            
            if key == ord('q'): 
                break
            elif key == ord('e'):
                active_evm_mode = "YIQ" if active_evm_mode == "BGR" else "BGR"
                video_pyr_initialized = False
                if videoPyramid is not None:
                    videoPyramid.fill(0.0)
            elif key == ord(' '):
                if not is_recording:
                    is_recording, record_frames = True, 0
                    sess_dir = os.path.join(OUTPUT_DIR, current_participant_id)
                    os.makedirs(sess_dir, exist_ok=True)
                    metadata_records = []
                    if video_writer is not None:
                        video_writer.release()
                    if SAVE_CAMERA_ACQUISITION:
                        video_writer = open_acquisition_video_writer(sess_dir, frame, fps)
            elif key == ord('n'):
                if is_recording:
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                
                if SAVE_METADATA and metadata_records:
                    flush_metadata_to_parquet(metadata_records, sess_dir)
                    metadata_records.clear()
                
                participant_counter += 1
                current_participant_id = f"P{participant_counter:03d}"
                
                buffers_initialized, video_pyr_initialized = False, False
                if videoPyramid is not None:
                    videoPyramid.fill(0.0)
                rr_history.clear(); spo2_history.clear(); bpm_all.clear()
                bpmBuffer[:] = np.nan

    except Exception:
        traceback.print_exc()
    finally:
        if SAVE_METADATA and metadata_records:
            flush_metadata_to_parquet(metadata_records, sess_dir)
        if video_writer is not None: 
            video_writer.release()
        if cap is not None: cap.release()
        if picam2 is not None: picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
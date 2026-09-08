import cv2
import numpy as np
import time
import traceback
from collections import deque
import mediapipe as mp
from scipy.signal import butter, filtfilt

# ==============================================================================
# DEVICE TOGGLE
# ==============================================================================
RUN_ON_LAPTOP = 1

# ==============================================================================
# CONFIGURATION & LAYOUT VARIABLES
# ==============================================================================
if RUN_ON_LAPTOP:
    realWidth, realHeight = 1280, 720
    PC_CAMERA_INDEX = 0
else:
    realWidth, realHeight = 640, 480
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

fps = 15.0

bufferSize = int(fps * 10)  
bpmBufferSize = 10
bpmCalcEvery = int(fps * 1) 

FACE_STABLE_SECONDS_REQUIRED = 5.0  

hr_low, hr_high = 0.7, 3.0   
rr_low, rr_high = 0.15, 0.5  
SPO2_A, SPO2_B = 110.0, 22.0

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

# Colors (BGR format)
BG_COLOR = (18, 18, 22)
CARD_COLOR = (30, 30, 38)
BORDER_COLOR = (55, 55, 68)
TITLE_COLOR = (140, 140, 160)
SUBTLE_TEXT = (100, 100, 120)
ACCENT_PULSE = (75, 75, 245)
ACCENT_BREATH = (245, 160, 60)
ACCENT_OXY = (60, 220, 120)
ACCENT_EVM = (200, 200, 0)

# ==============================================================================
# ROI MASKS
# ==============================================================================
SKIN_REGIONS = {
    "forehead": [10, 109, 67, 103, 54, 21, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 251, 284, 332, 297, 338]
}

EXCLUDE_REGIONS = {}

def get_segmented_mask(frame_shape, landmarks):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    landmarks_px = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) for lm in landmarks.landmark]
    def get_hull(indices):
        pts = np.array([landmarks_px[i] for i in indices], dtype=np.int32)
        return cv2.convexHull(pts)
    for region_name, indices in SKIN_REGIONS.items():
        cv2.fillPoly(mask, [get_hull(indices)], 255)
    for region_name, indices in EXCLUDE_REGIONS.items():
        cv2.fillPoly(mask, [get_hull(indices)], 0)
    return mask

def bbox_inside_roi(box, roi):
    if box is None or roi is None: return False
    bx, by, bw, bh = box
    rx1, ry1, rx2, ry2 = roi
    return (bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2)

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

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
            min_x, min_y = w, h
            max_x, max_y = 0, 0
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x + bw), max(max_y, y + bh)
            mp_face_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            
    return full_mask, mp_face_ok, mp_face_box, mask_contours


# ==============================================================================
# COLOR SPACES & SIGNAL PROCESSING
# ==============================================================================
M_bgr2yiq = np.array([[0.114, 0.587, 0.299], [-0.322, -0.274, 0.596], [0.312, -0.523, 0.211]], dtype=np.float32)
M_yiq2bgr = np.array([[1.703, -0.647, 1.0], [-1.106, -0.272, 1.0], [0.621, 0.956, 1.0]], dtype=np.float32)

def bgr_to_yiq(img_bgr):
    yiq = cv2.transform(img_bgr.astype(np.float32) / 255.0, M_bgr2yiq)
    return cv2.split(yiq)

def yiq_to_bgr(Y, I, Q):
    bgr = cv2.transform(cv2.merge([Y, I, Q]), M_yiq2bgr)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    if len(data) < 15:  
        return data
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if high >= 1.0: high = 0.99
    b, a = butter(order, [low, high], btype='band')
    padlen = min(len(data) - 1, 3 * max(len(b), len(a)))
    try: y = filtfilt(b, a, data, padlen=padlen)
    except ValueError: y = data 
    return y

def extract_pos(r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: int = 15) -> np.ndarray:
    N, l = len(r), int(2 * fps)
    if N < l: return np.zeros(N)
    
    r_win = np.lib.stride_tricks.sliding_window_view(r, l)
    g_win = np.lib.stride_tricks.sliding_window_view(g, l)
    b_win = np.lib.stride_tricks.sliding_window_view(b, l)
    
    eps = 1e-6
    rn = r_win / (np.mean(r_win, axis=1, keepdims=True) + eps)
    gn = g_win / (np.mean(g_win, axis=1, keepdims=True) + eps)
    bn = b_win / (np.mean(b_win, axis=1, keepdims=True) + eps)
    
    S1, S2 = gn - bn, gn + bn - 2 * rn
    alpha = np.std(S1, axis=1, keepdims=True) / (np.std(S2, axis=1, keepdims=True) + eps)
    h = S1 + alpha * S2
    h_zero_mean = h - np.mean(h, axis=1, keepdims=True)
    
    H = np.zeros(N)
    for i in range(l): H[i : i + h_zero_mean.shape[0]] += h_zero_mean[:, i]
    return H

def estimate_peak_bpm(signal: np.ndarray, fps_val: float, min_hz: float, max_hz: float) -> float:
    centered = signal - float(np.mean(signal))
    n_pad = 1024  
    mag = np.abs(np.fft.fft(centered, n=n_pad))
    freqs = np.fft.fftfreq(n_pad, d=1.0 / max(float(fps_val), 1e-6))
    
    band_mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(band_mask): return 0.0
    
    idx = int(np.argmax(mag * band_mask))
    return float(freqs[idx] * 60.0)

def robust_mean(data):
    if not data: return None
    data_list = list(data)
    if len(data_list) < 3: return float(np.mean(data_list))
    return float(np.median(data_list))


# ==============================================================================
# UI RENDERING
# ==============================================================================
def display_metric_value(face_detected_now: bool, vitals_enabled: bool, value):
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
    text_x, text_y = x + max(24, (w - tw) // 2), y + h // 2 + th // 2 + 10
    cv2.putText(canvas, val_str, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)

    if unit: cv2.putText(canvas, unit, (x + 24, y + h - 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, SUBTLE_TEXT, 1, cv2.LINE_AA)
    
    if show_pulse_icon and isinstance(current_bpm, (int, float, np.floating)) and current_bpm > 30:
        bps = float(current_bpm) / 60.0
        pulse_scale = 0.75 + 0.25 * np.sin((time.time() * bps * 2 * np.pi) % (2 * np.pi))
        heart_x, heart_y = x + w - 52, y + 56
        pts = (np.array([[0, -8], [8, -15], [15, -8], [0, 12], [-15, -8], [-8, -15]], np.int32) * pulse_scale * 1.4).astype(np.int32)
        cv2.fillPoly(canvas, [pts + [heart_x, heart_y]], color, lineType=cv2.LINE_AA)

def draw_camera_card(canvas, target_box, frame, evm_roi_bbox=None, mask_contours=None):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cam_h, cam_w = h - 20, w - 20
    
    actual_h, actual_w = frame.shape[:2]
    canvas[y + 10:y + 10 + cam_h, x + 10:x + 10 + cam_w] = cv2.resize(frame, (cam_w, cam_h))
    
    scale_x = cam_w / actual_w
    scale_y = cam_h / actual_h

    if evm_roi_bbox is not None:
        ex1, ey1, ex2, ey2 = evm_roi_bbox
        cv2.rectangle(canvas, (int(ex1 * scale_x) + x + 10, int(ey1 * scale_y) + y + 10), (int(ex2 * scale_x) + x + 10, int(ey2 * scale_y) + y + 10), ACCENT_EVM, 2, cv2.LINE_AA)
    
    if mask_contours is not None:
        scaled_contours = []
        for cnt in mask_contours:
            scaled_cnt = np.zeros_like(cnt)
            scaled_cnt[:, 0, 0] = cnt[:, 0, 0] * scale_x + x + 10
            scaled_cnt[:, 0, 1] = cnt[:, 0, 1] * scale_y + y + 10
            scaled_contours.append(scaled_cnt)
        cv2.polylines(canvas, scaled_contours, isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)


# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================
def main():
    cap = None
    picam2 = None
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
            picam2.configure(picam2.create_video_configuration(main={"size": (realWidth, realHeight), "format": "RGB888"}))
            picam2.start()
            time.sleep(2.0)
            mp_face_mesh = mp_face_mesh_module

        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        red_buffer, green_buffer, blue_buffer = [np.zeros((bufferSize,), dtype=np.float32) for _ in range(3)]
        bpmBuffer = np.full((bpmBufferSize,), np.nan, dtype=np.float32)
        bpm_all = []
        rr_history, spo2_history = deque(maxlen=10), deque(maxlen=10)
        
        buffers_initialized, video_pyr_initialized = False, False
        bufferIndex, bpmBufferIndex = 0, 0
        current_hr, current_rr, current_spo2 = None, None, None
        smoothed_bbox = None
        
        face_detected_now, vitals_enabled = False, False
        stable_face_start_time = None
        
        videoPyramid = None
        freqs = np.fft.fftfreq(bufferSize, d=1.0 / fps)
        mask = (freqs >= minFrequency) & (freqs <= maxFrequency)
        active_evm_mode = "BGR"
        
        cached_full_mask, cached_mp_face_ok, cached_mp_face_box, cached_mask_contours = None, False, None, []
        frame_counter = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL if RUN_ON_LAPTOP else cv2.WINDOW_AUTOSIZE)
        if RUN_ON_LAPTOP:
            cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)

        while True:
            loop_start = time.time()
            frame_counter += 1
            
            if RUN_ON_LAPTOP:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
            else:
                try:
                    frame = picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    time.sleep(0.05)
                    continue

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            current_h, current_w = frame.shape[:2]
            
            margin_x, margin_y = int(current_w * 0.05), int(current_h * 0.05)
            evm_roi_bbox = (margin_x, margin_y, current_w - margin_x, current_h - margin_y)

            if frame_counter % 2 != 0 or not cached_mp_face_ok:
                full_mask, mp_face_ok, mp_face_box, mask_contours = extract_mediapipe_roi(frame, face_mesh, scale=0.5)
                cached_full_mask, cached_mp_face_ok, cached_mp_face_box, cached_mask_contours = full_mask, mp_face_ok, mp_face_box, mask_contours
            else:
                full_mask, mp_face_ok, mp_face_box, mask_contours = cached_full_mask, cached_mp_face_ok, cached_mp_face_box, cached_mask_contours

            face_detected_now = mp_face_ok and bbox_inside_roi(mp_face_box, evm_roi_bbox)

            if face_detected_now:
                if stable_face_start_time is None: stable_face_start_time = time.time()
                stable_duration = time.time() - stable_face_start_time
                vitals_enabled = stable_duration >= FACE_STABLE_SECONDS_REQUIRED
            else:
                stable_face_start_time, stable_duration, vitals_enabled = None, 0.0, False

            vitals_roi, vitals_mask = None, None

            if mp_face_box is not None and face_detected_now:
                raw_x, raw_y, raw_w, raw_h = mp_face_box
                
                if smoothed_bbox is None or not face_detected_now:
                    smoothed_bbox = [raw_x, raw_y, raw_w, raw_h]
                else:
                    alpha_smooth = 0.15 
                    smoothed_bbox[0] = (1.0 - alpha_smooth) * smoothed_bbox[0] + alpha_smooth * raw_x
                    smoothed_bbox[1] = (1.0 - alpha_smooth) * smoothed_bbox[1] + alpha_smooth * raw_y
                    smoothed_bbox[2] = (1.0 - alpha_smooth) * smoothed_bbox[2] + alpha_smooth * raw_w
                    smoothed_bbox[3] = (1.0 - alpha_smooth) * smoothed_bbox[3] + alpha_smooth * raw_h

                vx1, vy1, vw, vh = [int(v) for v in smoothed_bbox]
                vx2, vy2 = clamp(vx1 + vw, 1, current_w), clamp(vy1 + vh, 1, current_h)
                vx1, vy1 = clamp(vx1, 0, current_w - 1), clamp(vy1, 0, current_h - 1)
                
                if vx2 > vx1 and vy2 > vy1:
                    vitals_roi = frame[vy1:vy2, vx1:vx2, :]
                    vitals_mask = full_mask[vy1:vy2, vx1:vx2]

            if vitals_roi is not None and vitals_mask is not None:
                if active_evm_mode == "YIQ":
                    Y, I, Q = bgr_to_yiq(vitals_roi)
                    processed_roi = cv2.merge([Y, I, Q])
                else:
                    processed_roi = vitals_roi.copy().astype(np.float32)

                gp = [processed_roi]
                for _ in range(levels):
                    gp.append(cv2.pyrDown(gp[-1]))
                    
                target_level = levels - 1
                size = (gp[target_level].shape[1], gp[target_level].shape[0])
                expanded = cv2.pyrUp(gp[target_level + 1], dstsize=size)
                laplacian = cv2.subtract(gp[target_level], expanded)

                if not video_pyr_initialized:
                    videoPyramid = np.zeros((bufferSize, laplacian.shape[0], laplacian.shape[1], 3), dtype=np.float32)
                    video_pyr_initialized = True
                else:
                    pyr_h, pyr_w = videoPyramid.shape[1], videoPyramid.shape[2]
                    if laplacian.shape[0] != pyr_h or laplacian.shape[1] != pyr_w:
                        laplacian = cv2.resize(laplacian, (pyr_w, pyr_h))

                videoPyramid[bufferIndex] = laplacian
                rolled_pyramid = np.roll(videoPyramid, -bufferIndex - 1, axis=0)

                fft_buffer = np.fft.fft(rolled_pyramid, axis=0)
                fft_buffer[~mask] = 0  
                filtered_buffer = np.real(np.fft.ifft(fft_buffer, axis=0))

                current_alpha = alpha_yiq if active_evm_mode == "YIQ" else alpha_bgr
                filtered_laplacian = filtered_buffer[-1] * current_alpha
                
                if active_evm_mode == "YIQ":
                    filtered_laplacian[:, :, 0] = 0.0 

                amplified_signal = filtered_laplacian
                for _ in range(target_level):
                    amplified_signal = cv2.pyrUp(amplified_signal)
                
                amplified_signal = cv2.resize(amplified_signal, (vitals_roi.shape[1], vitals_roi.shape[0]))

                binary_mask = (vitals_mask > 0).astype(np.float32)
                roi_mask_3d = np.expand_dims(binary_mask, axis=-1)
                masked_pulse = np.clip(amplified_signal * roi_mask_3d, -15.0, 15.0)

                if active_evm_mode == "YIQ":
                    out_vitals = processed_roi + masked_pulse
                    out_Y, out_I, out_Q = cv2.split(out_vitals)
                    out_vitals = yiq_to_bgr(out_Y, out_I, out_Q)
                else:
                    out_vitals = processed_roi + masked_pulse
                    out_vitals = np.clip(out_vitals, 0, 255).astype(np.uint8)

                # --- NEW COMPOSITING LOGIC ---
                # Expand the 2D mask to 3D so it aligns with the color channels
                mask_3d = np.expand_dims(vitals_mask > 0, axis=-1)
                original_roi = frame[vy1:vy2, vx1:vx2, :]
                
                # Apply the processed pixels ONLY inside the irregular mask boundaries
                frame[vy1:vy2, vx1:vx2, :] = np.where(mask_3d, out_vitals, original_roi)
                vitals_roi = frame[vy1:vy2, vx1:vx2, :]
                # -----------------------------

            # --- UPDATED EXTRACTION BLOCK ---
            if vitals_roi is not None and vitals_mask is not None:
                # Extract from the raw, unamplified frame copy
                raw_vitals_roi = raw_frame[vy1:vy2, vx1:vx2, :]
                mean_colors = cv2.mean(raw_vitals_roi, mask=vitals_mask)
                r_val, g_val, b_val = float(mean_colors[2]), float(mean_colors[1]), float(mean_colors[0])

                if r_val == 0.0 and g_val == 0.0 and b_val == 0.0 and buffers_initialized:
                    prev_idx = (bufferIndex - 1) % bufferSize
                    r_val, g_val, b_val = float(red_buffer[prev_idx]), float(green_buffer[prev_idx]), float(blue_buffer[prev_idx])

                if not buffers_initialized:
                    red_buffer[:], green_buffer[:], blue_buffer[:] = r_val, g_val, b_val
                    buffers_initialized = True
                else:
                    red_buffer[bufferIndex], green_buffer[bufferIndex], blue_buffer[bufferIndex] = r_val, g_val, b_val

            r_rolled = np.roll(red_buffer, -bufferIndex - 1)
            g_rolled = np.roll(green_buffer, -bufferIndex - 1)
            b_rolled = np.roll(blue_buffer, -bufferIndex - 1)

            if buffers_initialized and face_detected_now:
                raw_extracted_signal = extract_pos(r_rolled, g_rolled, b_rolled, int(fps))
                
                if vitals_enabled:
                    active_signal = bandpass_filter(raw_extracted_signal, hr_low, hr_high, fps, order=2)
                else:
                    active_signal = raw_extracted_signal

                if vitals_enabled and bufferIndex % bpmCalcEvery == 0:
                    bpm = estimate_peak_bpm(active_signal, fps, hr_low, hr_high)
                    if len(bpm_all) > 0:
                        last_bpm = bpm_all[-1]
                        max_change = 3.0 * (bpmCalcEvery / fps)
                        bpm = float(np.clip(bpm, last_bpm - max_change, last_bpm + max_change))
                        
                    bpmBuffer[bpmBufferIndex] = bpm
                    bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
                    bpm_all.append(bpm)
                    current_hr = np.nanmean(bpmBuffer)

                    r_rr_filtered = bandpass_filter(r_rolled, rr_low, rr_high, fps, order=2)
                    rr_raw = estimate_peak_bpm(r_rr_filtered, fps, rr_low, rr_high)
                    if 6 <= rr_raw <= 40: rr_history.append(rr_raw)
                    current_rr = robust_mean(rr_history)

                    r_hr_band = bandpass_filter(r_rolled, hr_low, hr_high, fps, order=2)
                    b_hr_band = bandpass_filter(b_rolled, hr_low, hr_high, fps, order=2)
                    
                    ac_r, dc_r = float(np.std(r_hr_band)), float(np.mean(r_rolled))
                    ac_b, dc_b = float(np.std(b_hr_band)), float(np.mean(b_rolled))

                    if dc_r > 1e-3 and dc_b > 1e-3 and ac_b > 1e-6:
                        ratio_of_ratios = (ac_r / dc_r) / (ac_b / dc_b)
                        spo2_raw = float(SPO2_A - (SPO2_B * ratio_of_ratios))
                        spo2_raw = clamp(spo2_raw, 95.0, 99.5)
                        spo2_history.append(spo2_raw)
                    current_spo2 = robust_mean(spo2_history)
            
            if not face_detected_now:
                current_hr, current_rr, current_spo2 = None, None, None
                buffers_initialized, video_pyr_initialized = False, False
                rr_history.clear(); spo2_history.clear(); bpm_all.clear()
                bpmBuffer[:] = np.nan

            canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
            pulse_display = display_metric_value(face_detected_now, vitals_enabled, current_hr)
            rr_display = display_metric_value(face_detected_now, vitals_enabled, current_rr)
            spo2_display = display_metric_value(face_detected_now, vitals_enabled, current_spo2)

            draw_card(canvas, BOX_PULSE, "HEART RATE", pulse_display, "BPM", ACCENT_PULSE, show_pulse_icon=True, current_bpm=current_hr if vitals_enabled else None)
            draw_card(canvas, BOX_BR, "RESPIRATION", rr_display, "BR/MIN", ACCENT_BREATH)
            draw_card(canvas, BOX_SPO2, "OXYGEN", spo2_display, "SpO2 %", ACCENT_OXY)
            draw_camera_card(canvas, BOX_CAMERA, frame, evm_roi_bbox=evm_roi_bbox, mask_contours=mask_contours)

            cv2.imshow(WINDOW_NAME, canvas)
            bufferIndex = (bufferIndex + 1) % bufferSize
            
            elapsed = time.time() - loop_start
            wait_ms = max(1, int(max(0.0, (1.0 / fps) - elapsed) * 1000))
            
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'): break
            elif key == ord('e'):
                active_evm_mode = "YIQ" if active_evm_mode == "BGR" else "BGR"
                video_pyr_initialized = False

    except Exception as e:
        print("\n" + "="*50)
        print("CRITICAL ERROR ENCOUNTERED!")
        print("="*50)
        traceback.print_exc()
        print("="*50)
        input("Press Enter to close this window...")
        
    finally:
        if cap is not None: cap.release()
        if picam2 is not None: picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
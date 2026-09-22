from __future__ import annotations

import os
import warnings

# =========================================================================
# Suppress Python Warnings (Protobuf deprecation UserWarnings)
# =========================================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================================================
# Standard & Third-Party Imports
# =========================================================================
import cv2
import numpy as np
import time
import datetime as dt
import traceback
import config
import json
import gui.main_window
from collections import deque
from queue import Empty

# =========================================================================
# Configuration & Custom Submodules
# =========================================================================
from config import *
from acquisitions.cam_capture import *
from roi.roi_manager import *
from roi.multi_roi import *
from signals.filtering import *
from signals.preprocessing import *
from methods.green import extract_green
from methods.chrom import extract_chrom
from methods.pos import extract_pos
from methods.ica_based import extract_ica
from methods.pca_based import extract_pca
from methods.pbv import extract_pbv
from methods.lgi import extract_lgi
from estimation.fft_hr import *
from estimation.sqi import *
from gui.main_window import *
from gui.plots import *
from data_logging.csv_logger import *
from data_logging.video_export import *
from data_logging.visualization import generate_session_plot

# ========================================
# Main Loop
# ========================================
def main():
    # --- Debug & Verbosity Toggles (1 = Enable, 0 = Disable) ---
    PRINT_CAMERA_SETTINGS   = 1  # Prints camera initialization metadata and hardware settings
    DEBUG_FACE_TRACKING     = 1  # Logs face detection status, target locking, and ROI stabilization
    DEBUG_VITALS            = 1  # Outputs real-time Heart Rate (HR), Respiration Rate (RR), and SpO2 calculations
    DEBUG_SQI               = 1  # Prints Signal Quality Index details (motion, brightness, periodicity)
    DEBUG_FPS               = 1  # Logs processing loop latency and the actual frames-per-second (FPS)
    DEBUG_STATE_MACHINE     = 1  # Tracks acquisition protocol phases, successful completions, and reset triggers
    DEBUG_BEAT_VIS          = 1  # Logs mathematical heartbeat frame captures and timing offsets
    USE_RAW_CAMERA_SETTINGS = 0  # 1 = Lock raw uncompressed settings (ISP off), 0 = Default auto settings

    # Setup Camera
    cap = cv2.VideoCapture(PC_CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
    cap.set(cv2.CAP_PROP_FPS, fps)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 forces Windows DirectShow to Auto-Exposure
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    # Conditionally apply raw uncompressed locks based on local toggle
    if USE_RAW_CAMERA_SETTINGS == 1:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)       # Adjusted to prevent overexposure
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        print("[INFO] Raw camera locks applied (ISP disabled).")

        if USE_PICAMERA2 and 'picam2' in globals():
            picam2.set_controls({
                "ExposureTime": 10000, 
                "AnalogueGain": 1.0, 
                "AwbEnable": False, 
                "ColourGains": (1.5, 1.2)
            }) # Haven't tested this yet
            print("[INFO] Picamera2 fixed controls applied.")
    else:
        # USE_RAW_CAMERA_SETTINGS = 0: Does nothing extra, acts completely like before
        print("[INFO] Using default camera settings (like before).")

    # Allow hardware to apply settings, then grab the metadata
    time.sleep(0.5)
    camera_settings = get_camera_metadata(cap)
    if PRINT_CAMERA_SETTINGS:
        print(f"[INFO] Camera initialized. Settings: {camera_settings}")

    # Initialize Explicit MediaPipe FaceMesh to prevent the .solutions AttributeError
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) # Haven't tried it with more than 1 face

    current_participant_id, _ = create_new_participant()
    
    # Track States and Configuration
    acquisition_active = ACQUISITION_ACTIVE_DEFAULT
    protocol_state = PROTOCOL_POSITIONING
    remaining_seconds = ACQUISITION_DURATION_SECONDS
    face_detected_now, vitals_enabled = False, False
    stable_face_start_time, vitals_unlocked_time = None, None
    last_reset_reason, last_completed_valid, last_completed_mean_sqi = "", None, None
    
    # M Key Multi-Method toggle
    RPPG_METHODS = {
    "GREEN": extract_green,
    "CHROM": extract_chrom,
    "POS": extract_pos,
    "ICA": extract_ica,
    "PCA": extract_pca,
    "PBV": extract_pbv,
    "LGI": extract_lgi
    }

    METHODS_CONFIG = [
        # --- Individual Methods ---
        ["GREEN"], 
        ["CHROM"], 
        ["POS"], 
        ["LGI"],
        ["ICA"], 
        ["PCA"],
        ["PBV"],
        # --- Method Combinations ---
        ["POS", "CHROM"], 
        ["LGI", "POS"],
        ["CHROM", "GREEN"],
        ["POS", "CHROM", "GREEN"],
        ["PBV", "CHROM"]
    ]
    current_method_idx = 6  # Defaults to ["PBV"]
    
    ROI_MODES = [
        # --- Individual ROIs ---
        ["FULL_FACE"],
        ["FULLFACE_USEFUL_AREA"], 
        ["FOREHEAD"], 
        ["LEFT_CHEEK"], 
        ["RIGHT_CHEEK"], 
        ["SEGMENTED"],
        # --- ROI Combinations ---
        ["LEFT_CHEEK", "RIGHT_CHEEK"],                           # Both Cheeks
        ["FOREHEAD", "LEFT_CHEEK", "RIGHT_CHEEK"]               # 3-Region Fusion
    ]
    current_roi_idx = 5  # Defaults to ["FOREHEAD"]
    EVM_MODES = ["BGR", "YIQ"]
    current_evm_idx = 1 # Defaults to ["YIQ"]

    # Buffer Initializations
    first_frame = np.zeros((videoHeight, videoWidth, videoChannels), dtype=np.uint8)
    Y_init, _, _ = bgr_to_yiq(first_frame)
    first_laplacian = build_laplacian_pyr(Y_init, levels)

    videoPyramid = [np.zeros((bufferSize, *level.shape), dtype=np.float32) for level in first_laplacian]
    fftAvg = np.zeros((bufferSize,), dtype=np.float32)

    red_buffer = np.zeros((bufferSize,), dtype=np.float32)
    blue_buffer = np.zeros((bufferSize,), dtype=np.float32)
    green_buffer = np.zeros((bufferSize,), dtype=np.float32)

    rr_buffer_size = int(fps * 20)
    long_raw_r = deque(maxlen=rr_buffer_size)
    long_raw_b = deque(maxlen=rr_buffer_size) 

    buffers_initialized = False
    video_pyr_initialized = False

    sqi_green_buffer, sqi_brightness_buffer = deque(maxlen=SQI_WINDOW), deque(maxlen=SQI_WINDOW)
    prev_roi_gray_for_sqi = None
    current_sqi_score, current_sqi_label = 0.0, "NO FACE"

    bpmBuffer = np.full((bpmBufferSize,), np.nan, dtype=np.float32)
    bpm_all = []
    rr_history, spo2_history = deque(maxlen=10), deque(maxlen=10)

    freqs = np.fft.fftfreq(bufferSize, d=1.0 / fps)
    mask = (freqs >= minFrequency) & (freqs <= maxFrequency)
    bufferIndex, bpmBufferIndex, frame_count = 0, 0, 0
    current_hr, current_rr, current_spo2 = None, None, None
    
    active_signal = np.zeros(bufferSize, dtype=np.float32)
    gui_waveform_buffer = deque([0.0] * bufferSize, maxlen=bufferSize)
    force_waveform_refresh = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
    cv2.moveWindow(WINDOW_NAME, 50, 50)

    # Logging State
    metadata_file, metadata_writer = None, None
    images_dir = None
    acquisition_video_writer = None
    acquisition_video_path = ""
    active_session_base = None

    workshop_stats = {
        "participants_created": 1, "completed_acquisitions": 0, "valid_acquisitions": 0,
        "low_quality_acquisitions": 0, "reset_acquisitions": 0, "completed_sqi_values": [],
        "last_participant_id": current_participant_id, "last_result": "CREATED", "last_mean_sqi": None,
    }
    write_workshop_stats_snapshot(workshop_stats)

    phase_elapsed, total_elapsed = 0.0, 0.0
    prev_loop_time = time.time()
    acquisition_start_time = None
    
    face_lost_start_time, poor_sqi_start_time = None, None
    acquisition_sqi_values, acquisition_hr_values, acquisition_rr_raw_values, acquisition_spo2_raw_values = [], [], [], []
    face_loss_events, face_loss_active = 0, False
    
    # NEW: Initialize the visualization state
    beat_vis_state = {"first_frame_captured": False, "target_second_frame": -1.0}

    smoothed_bbox = None

    try:
        while True:
            try:
                loop_start = time.time()

                if USE_PICAMERA2:
                    frame = picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue

                    if len(frame.shape) == 3 and frame.shape[2] == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)

                frame = cv2.resize(frame, (realWidth, realHeight))
                raw_frame_for_video = frame.copy()
                
                # EVM Target Bounding Box
                margin_x = int(realWidth * 0.05)   # 5% margin from edges
                margin_y = int(realHeight * 0.05)  # 5% margin from top/bottom

                x1, y1 = margin_x, margin_y
                x2, y2 = realWidth - margin_x, realHeight - margin_y

                det = frame[y1:y2, x1:x2, :]
                evm_roi_bbox = (x1, y1, x2, y2)

                # Process Mask
                active_roi_list = ROI_MODES[current_roi_idx]
                full_mask, mp_face_ok, mp_face_box, mask_contours = extract_combined_roi(frame, face_mesh, active_roi_list)
                face_detected_now = mp_face_ok and bbox_inside_roi(mp_face_box, evm_roi_bbox)
                
                if face_detected_now:
                    if stable_face_start_time is None: 
                        stable_face_start_time = time.time()
                        if DEBUG_FACE_TRACKING == 1: print("[FACE] Face detected. Locking...")
                    
                    stable_duration = time.time() - stable_face_start_time
                    prev_enabled = vitals_enabled
                    vitals_enabled = stable_duration >= FACE_STABLE_SECONDS_REQUIRED
                    
                    if vitals_enabled and not prev_enabled: 
                        vitals_unlocked_time = time.time()
                        force_waveform_refresh = True  # Instantly flush the raw noise and display the filtered wave
                        if DEBUG_FACE_TRACKING == 1: print("[FACE] Target locked. Vitals active.")
                else:
                    if vitals_enabled and DEBUG_FACE_TRACKING == 1: print("[FACE] Face lost.")
                    stable_face_start_time, stable_duration = None, 0.0
                    vitals_enabled, vitals_unlocked_time = False, None
                vitals_roi, vitals_mask, vitals_mask_float = None, None, None
                vitals_roi_bbox = None

                if mp_face_box is not None:
                    raw_x, raw_y, raw_w, raw_h = mp_face_box
                    
                    # Apply Exponential Moving Average (EMA) to smooth the box
                    if smoothed_bbox is None or not face_detected_now:
                        smoothed_bbox = [raw_x, raw_y, raw_w, raw_h]
                    else:
                        # 15% new frame data, 85% previous frame data (hyperparameter can be adjusted for responsiveness vs stability)
                        alpha_smooth = 0.15 
                        smoothed_bbox[0] = (1.0 - alpha_smooth) * smoothed_bbox[0] + alpha_smooth * raw_x
                        smoothed_bbox[1] = (1.0 - alpha_smooth) * smoothed_bbox[1] + alpha_smooth * raw_y
                        smoothed_bbox[2] = (1.0 - alpha_smooth) * smoothed_bbox[2] + alpha_smooth * raw_w
                        smoothed_bbox[3] = (1.0 - alpha_smooth) * smoothed_bbox[3] + alpha_smooth * raw_h

                    # Unpack the smoothed integers
                    vx1, vy1, vw, vh = [int(v) for v in smoothed_bbox]
                    
                    vx2, vy2 = clamp(vx1 + vw, 1, realWidth), clamp(vy1 + vh, 1, realHeight)
                    vx1, vy1 = clamp(vx1, 0, realWidth - 1), clamp(vy1, 0, realHeight - 1)
                    if vx2 > vx1 and vy2 > vy1:
                        vitals_roi = frame[vy1:vy2, vx1:vx2, :]
                        raw_mask = full_mask[vy1:vy2, vx1:vx2]
                    
                        # Generate the soft probabilistic mask
                        vitals_mask, vitals_mask_float = generate_soft_mask(raw_mask)
                        vitals_roi_bbox = (vx1, vy1, vx2, vy2)
                else:
                    smoothed_bbox = None
                # Run SQI
                current_motion_score = 0.0
                current_brightness_score = 0.0
                current_periodicity_score = 0.0

                if face_detected_now and vitals_roi is not None:
                    try:
                        roi_gray_for_sqi = cv2.cvtColor(vitals_roi, cv2.COLOR_BGR2GRAY)
                        current_motion_score = compute_motion_score(prev_roi_gray_for_sqi, roi_gray_for_sqi)
                        prev_roi_gray_for_sqi = roi_gray_for_sqi.copy()

                        roi_brightness = float(np.mean(roi_gray_for_sqi))
                        sqi_brightness_buffer.append(roi_brightness)
                        current_brightness_score = compute_brightness_score(sqi_brightness_buffer)

                        mean_colors_sqi = cv2.mean(vitals_roi, mask=vitals_mask)
                        current_green_mean = float(mean_colors_sqi[1])
                        sqi_green_buffer.append(current_green_mean)
                        
                        current_periodicity_score = compute_periodicity_score(sqi_green_buffer, fps)
                        current_sqi_score, current_sqi_label = compute_signal_quality_index(
                            current_motion_score, current_brightness_score, current_periodicity_score, face_detected_now, vitals_enabled
                        )
                        
                        if DEBUG_SQI == 1 and vitals_enabled and (frame_count % int(fps) == 0):
                            print(f"[SQI] Score: {current_sqi_score:.1f} ({current_sqi_label}) | M: {current_motion_score:.2f} | B: {current_brightness_score:.2f} | P: {current_periodicity_score:.2f}")

                    except Exception as e:
                        current_sqi_score, current_sqi_label = 0.0, "POOR SIGNAL"
                else:
                    prev_roi_gray_for_sqi = None
                    if not face_detected_now: current_sqi_score, current_sqi_label = 0.0, "NO FACE"


                # -----------------------------------------------------------------
                # APPLY DYNAMIC ROI EVM (GAUSSIAN PYRAMID + YIQ TOGGLE)
                if vitals_roi is not None and vitals_mask is not None:
                    active_evm_mode = EVM_MODES[current_evm_idx]
                    
                    # A. Convert to YIQ if active
                    if active_evm_mode == "YIQ":
                        Y, I, Q = bgr_to_yiq(vitals_roi)
                        processed_roi = cv2.merge([Y, I, Q])
                    else:
                        processed_roi = vitals_roi.copy().astype(np.float32)

                    # B. Gaussian Spatial Blur (cv2.pyrDown)
                    blurred = processed_roi
                    for _ in range(levels):
                        blurred = cv2.pyrDown(blurred)

                    # Lock dimensions for temporal buffer
                    if not video_pyr_initialized:
                        videoPyramid = np.zeros((bufferSize, blurred.shape[0], blurred.shape[1], 3), dtype=np.float32)
                        video_pyr_initialized = True
                    else:
                        pyr_h, pyr_w = videoPyramid.shape[1], videoPyramid.shape[2]
                        if blurred.shape[0] != pyr_h or blurred.shape[1] != pyr_w:
                            blurred = cv2.resize(blurred, (pyr_w, pyr_h))

                    videoPyramid[bufferIndex] = blurred
                    rolled_pyramid = np.roll(videoPyramid, -bufferIndex - 1, axis=0)

                    # C. Temporal FFT (Only runs ONCE per frame!)
                    fft_buffer = np.fft.fft(rolled_pyramid, axis=0)
                    fft_buffer[~mask] = 0  
                    filtered_buffer = np.real(np.fft.ifft(fft_buffer, axis=0))

                    # D. Apply Dynamic Alpha from config.py based on Mode
                    if active_evm_mode == "YIQ":
                        current_alpha = alpha_yiq
                    else:
                        current_alpha = alpha_bgr
                        
                    filtered_frame = filtered_buffer[-1] * current_alpha
                    
                    # E. ZERO OUT LUMINANCE (The YIQ step to block lighting noise)
                    if active_evm_mode == "YIQ":
                        filtered_frame[:, :, 0] = 0.0

                    # F. Scale back up (cv2.pyrUp)
                    for _ in range(levels):
                        filtered_frame = cv2.pyrUp(filtered_frame)
                    filtered_frame = cv2.resize(filtered_frame, (vitals_roi.shape[1], vitals_roi.shape[0]))

                    # G. Mask and Clamp Shockwaves using the soft gradient
                    roi_mask_3d = np.expand_dims(vitals_mask_float, axis=-1)
                    masked_pulse = np.clip(filtered_frame * roi_mask_3d, -EVM_CLIP_LIMIT, EVM_CLIP_LIMIT)

                    # H. Add back and convert to BGR
                    if active_evm_mode == "YIQ":
                        out_vitals = processed_roi + masked_pulse
                        out_Y, out_I, out_Q = cv2.split(out_vitals)
                        out_vitals = yiq_to_bgr(out_Y, out_I, out_Q)
                    else:
                        out_vitals = processed_roi + masked_pulse
                        out_vitals = np.clip(out_vitals, 0, 255).astype(np.uint8)

                    # Overwrite so the next block extracts from the amplified image
                    frame[vy1:vy2, vx1:vx2, :] = out_vitals
                    vitals_roi = out_vitals

                # -----------------------------------------------------------------
                # Extract Color signals (Split Pipeline)
                if vitals_roi is not None and vitals_mask_float is not None:
                    # TRACK 1: HR & UI (From the EVM amplified 'out_vitals')
                    b_val, g_val, r_val = compute_weighted_mean(out_vitals, vitals_mask_float)
                    
                    # TRACK 2: SpO2 & RR (From the pure 'raw_frame_for_video')
                    vx1, vy1, vx2, vy2 = vitals_roi_bbox
                    raw_roi = raw_frame_for_video[vy1:vy2, vx1:vx2, :]
                    raw_b_val, _, raw_r_val = compute_weighted_mean(raw_roi, vitals_mask_float)

                    # Feed the 20-second queues
                    long_raw_r.append(raw_r_val)
                    long_raw_b.append(raw_b_val)

                    if r_val == 0.0 and g_val == 0.0 and b_val == 0.0 and buffers_initialized:
                        prev_idx = (bufferIndex - 1) % bufferSize
                        r_val = float(red_buffer[prev_idx])
                        g_val = float(green_buffer[prev_idx])
                        b_val = float(blue_buffer[prev_idx])

                    if not buffers_initialized:
                        red_buffer[:] = r_val
                        green_buffer[:] = g_val
                        blue_buffer[:] = b_val
                        buffers_initialized = True
                    else:
                        red_buffer[bufferIndex] = r_val
                        green_buffer[bufferIndex] = g_val
                        blue_buffer[bufferIndex] = b_val

                r_rolled = np.roll(red_buffer, -bufferIndex - 1)
                g_rolled = np.roll(green_buffer, -bufferIndex - 1)
                b_rolled = np.roll(blue_buffer, -bufferIndex - 1)

                active_methods = METHODS_CONFIG[current_method_idx]
                current_method_name = "+".join(active_methods)
                
                extracted_signals = []
                for m_name in active_methods:
                    extraction_function = RPPG_METHODS[m_name]
                    
                    # Only pass fps to the window-based methods that require it
                    if m_name in ["CHROM", "POS", "PBV"]:
                        sig = extraction_function(r_rolled, g_rolled, b_rolled, fps=fps)
                    else:
                        # For GREEN, ICA, PCA, etc.
                        sig = extraction_function(r_rolled, g_rolled, b_rolled)
                        
                    extracted_signals.append(sig)
                
                # Fuse the signals together
                raw_extracted_signal = fuse_rppg_signals(extracted_signals)

                # 1. Calculate the mathematically pure window for FFT and HR estimation
                if vitals_enabled:
                    active_signal = bandpass_filter(raw_extracted_signal, hr_low, hr_high, fps, order=5)
                else:
                    active_signal = raw_extracted_signal

                # 2. Build the stitched buffer strictly for the UI to prevent visual bouncing
                if force_waveform_refresh:
                    gui_waveform_buffer.extend(float(val) for val in active_signal)
                    force_waveform_refresh = False
                else:
                    gui_waveform_buffer.append(float(active_signal[-1]))
                
                display_waveform = np.array(gui_waveform_buffer, dtype=np.float32)


                # Gate frequency analysis behind stabilization requirement
                if vitals_enabled:
                    signal_centered = active_signal - np.mean(active_signal)
                    window = np.hanning(len(signal_centered))
                    signal_fft = np.abs(np.fft.fft(signal_centered * window))
                    
                    # Apply EMA smoothing to the spectrum bins to stop frantic updating
                    alpha_spectrum = 0.2  # 20% new data, 80% historical smoothing (hyperparameter can be adjusted)
                    for b in range(bufferSize): 
                        fftAvg[b] = (alpha_spectrum * signal_fft[b]) + ((1.0 - alpha_spectrum) * fftAvg[b])

                    if bufferIndex % bpmCalcEvery == 0:
                        # 1. Isolate the physiological frequency band
                        hr_mask = (freqs >= hr_low) & (freqs <= hr_high)
                        
                        if np.any(hr_mask):
                            masked_spectrum = fftAvg * hr_mask
                            peak_index = int(np.argmax(masked_spectrum))
                            
                            # 2. Sub-bin Parabolic Interpolation for continuous precision
                            if 0 < peak_index < len(masked_spectrum) - 1:
                                alpha = masked_spectrum[peak_index - 1]
                                beta = masked_spectrum[peak_index]
                                gamma = masked_spectrum[peak_index + 1]
                                
                                # Calculate the fractional offset [-0.5 to 0.5]
                                denominator = alpha - 2 * beta + gamma
                                p = 0.5 * (alpha - gamma) / denominator if denominator != 0 else 0.0
                                
                                # Apply offset to exact frequency
                                exact_freq = freqs[peak_index] + p * (freqs[1] - freqs[0])
                            else:
                                exact_freq = freqs[peak_index]
                                
                            bpm = float(exact_freq * 60.0)
                        else:
                            bpm = 0.0
                            
                        # 3. SLEW RATE LIMITER (3 BPM / sec constraint) (Found it in bibliography)
                        if len(bpm_all) > 0 and bpm > 0.0:
                            last_bpm = bpm_all[-1]
                            max_change = 3.0 * (bpmCalcEvery / fps)
                            bpm = float(np.clip(bpm, last_bpm - max_change, last_bpm + max_change))
                        
                        # 4. Add to rolling buffers
                        if bpm > 0.0:
                            bpmBuffer[bpmBufferIndex] = bpm
                            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
                            bpm_all.append(bpm)
                        
                        # 5. Final output is smoothly averaged
                        current_hr = np.nanmean(bpmBuffer)
                        rr_mask = (freqs >= rr_low) & (freqs <= rr_high)

                        # Only calculate slow vitals if we have at least 10 seconds of history
                        if len(long_raw_r) >= int(fps * 10):
                            r_long_array = np.array(long_raw_r, dtype=np.float32)
                            b_long_array = np.array(long_raw_b, dtype=np.float32)

                            # 1. Dedicated Respiratory Filter (0.15 - 0.5 Hz)
                            r_rr_filtered = bandpass_filter(r_long_array, rr_low, rr_high, fps, order=2) # The order can be adjusted for responsiveness vs smoothness
                            rr_raw = estimate_peak_bpm(r_rr_filtered, fps, rr_low, rr_high)

                            if 6 <= rr_raw <= 40:
                                if len(rr_history) > 0:
                                    last_rr = rr_history[-1]
                                    rr_raw = float(np.clip(rr_raw, last_rr - 2.0, last_rr + 2.0))
                                rr_history.append(rr_raw)
                            
                            rr_target = robust_mean(rr_history)
                            if rr_target is not None:
                                current_rr = rr_target if current_rr is None else (0.2 * rr_target) + (0.8 * current_rr)

                            # 2. Dedicated Cardiac Filter for SpO2 AC Modulation
                            r_hr_band = bandpass_filter(r_long_array, hr_low, hr_high, fps, order=2) # The order can be adjusted for responsiveness vs smoothness
                            b_hr_band = bandpass_filter(b_long_array, hr_low, hr_high, fps, order=2) # The order can be adjusted for responsiveness vs smoothness

                            ac_r, dc_r = float(np.std(r_hr_band)), float(np.mean(r_long_array))
                            ac_b, dc_b = float(np.std(b_hr_band)), float(np.mean(b_long_array))

                            if dc_r > 1e-3 and dc_b > 1e-3 and ac_b > 1e-6:
                                ratio_of_ratios = (ac_r / dc_r) / (ac_b / dc_b)
                                spo2_raw = float(np.clip(SPO2_A - (SPO2_B * ratio_of_ratios), 85.0, 100.0))
                                
                                if 85.0 <= spo2_raw <= 100.0:
                                    if len(spo2_history) > 0:
                                        last_spo2 = spo2_history[-1]
                                        spo2_raw = float(np.clip(spo2_raw, last_spo2 - 1.0, last_spo2 + 1.0))
                                    spo2_history.append(spo2_raw)

                            spo2_target = robust_mean(spo2_history)
                            if spo2_target is not None:
                                current_spo2 = spo2_target if current_spo2 is None else (0.5 * spo2_target) + (0.5 * current_spo2) # Hyperparameter can be adjusted
                        
                        if DEBUG_VITALS == 1:
                            h_val = f"{current_hr:.1f}" if current_hr is not None else "N/A"
                            r_val = f"{current_rr:.1f}" if current_rr is not None else "N/A"
                            s_val = f"{current_spo2:.1f}" if current_spo2 is not None else "N/A"
                            print(f"[VITALS] HR: {h_val} BPM | RR: {r_val} Br/min | SpO2: {s_val}%")
                else: 
                    fftAvg[:] = 0
                    current_hr, current_rr, current_spo2 = None, None, None


                if not face_detected_now:
                    current_hr, current_rr, current_spo2 = None, None, None
                    buffers_initialized, video_pyr_initialized = False, False
                    rr_history.clear(); spo2_history.clear(); bpm_all.clear()
                    bpmBuffer[:] = np.nan; sqi_green_buffer.clear(); sqi_brightness_buffer.clear()
                    gui_waveform_buffer.extend([0.0] * bufferSize)

                # Process State Machine for Acquisition Protocols & Exports
                if acquisition_active and protocol_state == PROTOCOL_LOCKING:
                    phase_elapsed, total_elapsed, remaining_seconds = 0.0, 0.0, ACQUISITION_DURATION_SECONDS
                    if vitals_enabled:
                        protocol_state = PROTOCOL_ACQUISITION
                        acquisition_start_time = time.time()
                        phase_elapsed, total_elapsed, remaining_seconds = 0.0, 0.0, ACQUISITION_DURATION_SECONDS
                        face_lost_start_time, poor_sqi_start_time = None, None
                        acquisition_sqi_values, acquisition_hr_values, acquisition_rr_raw_values, acquisition_spo2_raw_values = [], [], [], []
                        face_loss_events, face_loss_active = 0, False
                        last_reset_reason, last_completed_valid, last_completed_mean_sqi = "", None, None
                        
                        # Reset visualization state for the new acquisition
                        beat_vis_state = {"first_frame_captured": False, "target_second_frame": -1.0}

                        # DO NOT touch buffers_initialized or video_pyr_initialized here!
                        # This ensures the running EVM filters keep their active history.

                        close_participant_session_files(metadata_file)

                        active_session_base, images_dir, metadata_file, metadata_writer = open_participant_session_files(current_participant_id)
                        acquisition_video_writer, acquisition_video_path = open_acquisition_video_writer(current_participant_id, realWidth, realHeight, actual_fps if 'actual_fps' in locals() and actual_fps > 0 else fps)
                elif acquisition_active and protocol_state == PROTOCOL_ACQUISITION and acquisition_start_time is not None:
                    total_elapsed = time.time() - acquisition_start_time
                    phase_elapsed = total_elapsed
                    remaining_seconds = max(0.0, ACQUISITION_DURATION_SECONDS - phase_elapsed)

                    acquisition_sqi_values.append(float(current_sqi_score))
                    if current_hr is not None: acquisition_hr_values.append(float(current_hr))
                    if current_rr is not None: acquisition_rr_raw_values.append(float(current_rr))
                    if current_spo2 is not None: acquisition_spo2_raw_values.append(float(current_spo2))

                    now_for_validity = time.time()

                    if not face_detected_now:
                        if not face_loss_active:
                            face_loss_events += 1
                            face_loss_active = True
                        if face_lost_start_time is None: face_lost_start_time = now_for_validity
                    else: face_loss_active, face_lost_start_time = False, None

                    if current_sqi_score < POOR_SQI_THRESHOLD:
                        if poor_sqi_start_time is None: poor_sqi_start_time = now_for_validity
                    else: poor_sqi_start_time = None

                    reset_reason = ""
                    if face_lost_start_time is not None and (now_for_validity - face_lost_start_time) >= FACE_LOSS_RESET_SECONDS:
                        reset_reason = f"Face lost for more than {FACE_LOSS_RESET_SECONDS:.0f}s. Please reposition and repeat."
                    elif poor_sqi_start_time is not None and (now_for_validity - poor_sqi_start_time) >= POOR_SQI_RESET_SECONDS:
                        reset_reason = f"Signal quality remained below {POOR_SQI_THRESHOLD:.0f}/100 for more than {POOR_SQI_RESET_SECONDS:.0f}s. Please repeat."

                    if reset_reason:
                        if DEBUG_STATE_MACHINE == 1: print(f"[STATE] Reset: {reset_reason}")
                        close_participant_session_files(metadata_file)
                        close_acquisition_video_writer(acquisition_video_writer)
                        acquisition_video_writer, acquisition_video_path = None, ""
                        acquisition_active = False
                        protocol_state = PROTOCOL_POSITIONING
                        acquisition_start_time = None
                        phase_elapsed, total_elapsed, remaining_seconds = 0.0, 0.0, ACQUISITION_DURATION_SECONDS

                        stable_face_start_time, vitals_enabled, vitals_unlocked_time = None, False, None
                        face_lost_start_time, poor_sqi_start_time, acquisition_sqi_values = None, None, []
                        last_reset_reason, last_completed_valid, last_completed_mean_sqi = reset_reason, None, None

                        write_acquisition_summary(
                            current_participant_id, status="RESET", mean_sqi=safe_numeric_mean(acquisition_sqi_values), duration_saved=phase_elapsed,
                            mean_hr=safe_numeric_mean(acquisition_hr_values), median_hr=safe_numeric_median(acquisition_hr_values),
                            mean_rr_raw=safe_numeric_mean(acquisition_rr_raw_values), mean_spo2_raw=safe_numeric_mean(acquisition_spo2_raw_values),
                            face_loss_events=face_loss_events, reset_reason=reset_reason,
                        )
                        workshop_stats["reset_acquisitions"] += 1
                        workshop_stats["last_participant_id"] = current_participant_id
                        workshop_stats["last_result"] = "RESET"
                        workshop_stats["last_mean_sqi"] = safe_numeric_mean(acquisition_sqi_values)
                        append_workshop_event(current_participant_id, "ACQUISITION_RESET", "RESET", workshop_stats["last_mean_sqi"], reset_reason)
                        write_workshop_stats_snapshot(workshop_stats)

                    elif phase_elapsed >= ACQUISITION_DURATION_SECONDS:
                        if DEBUG_STATE_MACHINE == 1: print("[STATE] Acquisition Complete.")
                        acquisition_active = False
                        protocol_state = PROTOCOL_COMPLETE
                        remaining_seconds = 0.0

                        mean_sqi = float(np.mean(acquisition_sqi_values)) if acquisition_sqi_values else 0.0
                        last_completed_mean_sqi = mean_sqi
                        last_completed_valid = mean_sqi >= VALID_MEAN_SQI_THRESHOLD
                        last_reset_reason = ""

                        # Capture the folder path before closing files
                        completed_folder = create_participant_folder(current_participant_id)
                        metadata_csv_target = os.path.join(completed_folder, "metadata.csv")
                        
                        # Save Camera Hardware Settings
                        camera_config_target = os.path.join(completed_folder, "hardware_config.json")
                        with open(camera_config_target, 'w') as f:
                            json.dump(camera_settings, f, indent=4)
                        # ------------------------------------------

                        close_participant_session_files(metadata_file)
                        close_acquisition_video_writer(acquisition_video_writer)
                        
                        # Generate the visualization plot by reading the completed CSV
                        generate_session_plot(completed_folder, metadata_csv_target)

                        acquisition_video_writer, acquisition_video_path = None, ""
                        metadata_file, metadata_writer, active_session_base, images_dir = None, None, None, None

                        stable_face_start_time, vitals_enabled, vitals_unlocked_time = None, False, None
                        face_lost_start_time, poor_sqi_start_time = None, None

                        status_text = "VALID" if last_completed_valid else "LOW QUALITY"
                        write_acquisition_summary(
                            current_participant_id, status=status_text, mean_sqi=mean_sqi, duration_saved=ACQUISITION_DURATION_SECONDS,
                            mean_hr=safe_numeric_mean(acquisition_hr_values), median_hr=safe_numeric_median(acquisition_hr_values),
                            mean_rr_raw=safe_numeric_mean(acquisition_rr_raw_values), mean_spo2_raw=safe_numeric_mean(acquisition_spo2_raw_values),
                            face_loss_events=face_loss_events, reset_reason="", acquisition_video_path=acquisition_video_path,
                        )

                        workshop_stats["completed_acquisitions"] += 1
                        if last_completed_valid:
                            workshop_stats["valid_acquisitions"] += 1
                            workshop_result = "VALID"
                        else:
                            workshop_stats["low_quality_acquisitions"] += 1
                            workshop_result = "LOW QUALITY"

                        workshop_stats["completed_sqi_values"].append(mean_sqi)
                        workshop_stats["last_participant_id"] = current_participant_id
                        workshop_stats["last_result"] = workshop_result
                        workshop_stats["last_mean_sqi"] = mean_sqi

                        event_msg = f"Completed 30s acquisition. Mean SQI={mean_sqi:.1f}. Valid={last_completed_valid}"
                        append_workshop_event(current_participant_id, "ACQUISITION_COMPLETE", workshop_result, mean_sqi, event_msg)
                        write_workshop_stats_snapshot(workshop_stats)

                elif protocol_state == PROTOCOL_POSITIONING:
                    phase_elapsed, total_elapsed, remaining_seconds = 0.0, 0.0, ACQUISITION_DURATION_SECONDS

                # Handle continuous logging requests based on Protocol
                # -----------------------------------------------------------------
                # UNIFIED LOGGING ARCHITECTURE
                # -----------------------------------------------------------------
                if acquisition_active and protocol_state == PROTOCOL_ACQUISITION:
                    
                    # 1. Video Writer
                    if acquisition_video_writer is not None:
                        try: 
                            acquisition_video_writer.write(raw_frame_for_video.copy())
                        except Exception: 
                            pass

                    # 2. Extract ML Images and grab their Foreign Key paths (Using EVM Amplified Frame)
                    roi_image_paths_str = ""
                    if mp_face_box is not None:
                       candidate_rois = extract_candidate_rppg_rois(frame, mp_face_box)
                       roi_image_paths_str = save_ml_roi_images(images_dir, current_participant_id, frame_count, candidate_rois, frame)

                    # Process mathematical heartbeat frame extractions
                    valid_hr = current_hr if current_hr is not None and not np.isnan(current_hr) else 0.0
                    
                    beat_vis_state = process_beat_visualizations(
                        beat_vis_state,
                        phase_elapsed,       # <--- Pass pure time in seconds
                        valid_hr,
                        images_dir,
                        raw_frame_for_video, # The untouched original
                        frame,               # The EVM amplified frame
                        full_mask,           # The MediaPipe mask
                        debug=DEBUG_BEAT_VIS
                    )

                    # 3. Compile and write the unified Metadata row
                    if metadata_writer is not None and metadata_file is not None:
                        try:
                            f_x1, f_y1, f_x2, f_y2 = ("", "", "", "")
                            if vitals_roi_bbox is not None: 
                                f_x1, f_y1, f_x2, f_y2 = vitals_roi_bbox

                            rgb_vals = ("", "", "")
                            brightness_mean, brightness_std = "", ""
                            
                            if vitals_roi is not None and vitals_mask is not None:
                                mean_bgr = cv2.mean(vitals_roi, mask=vitals_mask)
                                rgb_vals = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

                                gray_roi = cv2.cvtColor(vitals_roi, cv2.COLOR_BGR2GRAY)
                                valid_gray = gray_roi[vitals_mask > 0]
                                if len(valid_gray) > 0:
                                    brightness_mean = float(np.mean(valid_gray))
                                    brightness_std = float(np.std(valid_gray))

                            now_loop = time.time()
                            actual_fps = 1.0 / max(now_loop - prev_loop_time, 1e-6)
                            prev_loop_time = now_loop
                            
                            if DEBUG_FPS == 1 and (frame_count % int(fps) == 0):
                                print(f"[PERF] Target: {fps} FPS | Actual: {actual_fps:.1f} FPS")
                            row_data = [
                                dt.datetime.now().isoformat(), current_participant_id, frame_count, 
                                int(face_detected_now), int(vitals_enabled), protocol_state,
                                f_x1, f_y1, f_x2, f_y2,
                                round(rgb_vals[0], 3) if rgb_vals[0] != "" else "", 
                                round(rgb_vals[1], 3) if rgb_vals[1] != "" else "", 
                                round(rgb_vals[2], 3) if rgb_vals[2] != "" else "",
                                round(brightness_mean, 3) if brightness_mean != "" else "", 
                                round(brightness_std, 3) if brightness_std != "" else "", 
                                round(float(actual_fps), 3),
                                round(float(current_hr), 2) if current_hr is not None else "",
                                round(float(current_rr), 2) if current_rr is not None else "", 
                                round(float(current_spo2), 2) if current_spo2 is not None else "",
                                current_method_name,
                                round(float(active_signal[-1]), 4) if 'active_signal' in locals() and len(active_signal) > 0 else "",
                                round(float(current_sqi_score), 2), current_sqi_label, 
                                round(float(current_motion_score), 2), round(float(current_brightness_score), 2),
                                round(float(current_periodicity_score), 2), 
                                round(float(phase_elapsed), 3), round(float(total_elapsed), 3),
                                roi_image_paths_str
                            ]
                            
                            metadata_writer.writerow(row_data)
                            metadata_file.flush()
                        except Exception as e:
                            print(f"Metadata write error: {e}")

                # Fade mapping
                fade_alpha = 1.0
                if vitals_enabled and vitals_unlocked_time is not None:
                    fade_alpha = min(1.0, max(0.0, (time.time() - vitals_unlocked_time) / FADE_IN_SECONDS))

                pulse_display = display_metric_value(face_detected_now, vitals_enabled, apply_fade(current_hr, fade_alpha))
                rr_display = display_metric_value(face_detected_now, vitals_enabled, apply_fade(current_rr, fade_alpha))
                spo2_display = display_metric_value(face_detected_now, vitals_enabled, apply_fade(current_spo2, fade_alpha))

                status_items = build_status_items(
                    face_detected_now, vitals_enabled, stable_duration, acquisition_active, 
                    current_sqi_label, current_sqi_score, protocol_state, remaining_seconds, 
                    last_reset_reason, last_completed_valid, last_completed_mean_sqi
                )

                # Execute rendering
                canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
                draw_card(canvas, BOX_PULSE, "HEART RATE", pulse_display, "BPM", ACCENT_PULSE, show_pulse_icon=True, current_bpm=current_hr if vitals_enabled else None)
                draw_waveform_card(canvas, BOX_WAVE, display_waveform, current_method_name)
                draw_protocol_timer_card(canvas, BOX_EMOTION, protocol_state, remaining_seconds)
                draw_status_card(canvas, BOX_STATUS, status_items)
                
                # Render Spectrum replacing the Logo spot as in previous iteration
                draw_logo_card(canvas, BOX_LOGO, None, freqs, fftAvg, minFrequency, maxFrequency, current_hr)
                
                draw_camera_card(canvas, BOX_CAMERA, frame, evm_roi_bbox=evm_roi_bbox, mask_contours=mask_contours)
                draw_card(canvas, BOX_BR, "RESPIRATION", rr_display, "BR/MIN", ACCENT_BREATH)
                draw_card(canvas, BOX_SPO2, "OXYGEN", spo2_display, "SpO2 %", ACCENT_OXY)
                draw_acquisition_control(canvas, acquisition_active, EVM_MODES[current_evm_idx])
                draw_participant_id(canvas, current_participant_id)

                cv2.imshow(WINDOW_NAME, canvas)
                
                # Calculate the required delay to maintain FPS here, instead of at the end
                elapsed = time.time() - loop_start
                wait_ms = max(1, int(max(0.0, (1.0 / fps) - elapsed) * 1000))
                
                # Listen for the key for the entire duration of the wait_ms
                key = cv2.waitKey(wait_ms) & 0xFF
                
                # Key Handlers
                if key == ord('q'): break
                
                elif key == ord('r'):
                    current_roi_idx = (current_roi_idx + 1) % len(ROI_MODES)
                elif key == ord('m'):
                    current_method_idx = (current_method_idx + 1) % len(METHODS_CONFIG)
                    force_waveform_refresh = True
                elif key == ord(' '):
                    if acquisition_active:
                        acquisition_active, protocol_state = False, PROTOCOL_POSITIONING
                        close_participant_session_files(metadata_file)
                        close_acquisition_video_writer(acquisition_video_writer)
                        acquisition_video_writer, acquisition_video_path = None, ""
                        metadata_file, metadata_writer, active_session_base, images_dir = None, None, None, None
                    else:
                        acquisition_active, protocol_state = True, PROTOCOL_LOCKING
                        
                        # Reset visualization state for the new run
                        beat_vis_state = {"first_frame_captured": False, "target_second_frame": -1.0}
                        prev_loop_time = time.time()
                        smoothed_bbox = None
                        
                        # Open the logging files cleanly without wiping your live signal
                        active_session_base, images_dir, metadata_file, metadata_writer = open_participant_session_files(current_participant_id)
                        acquisition_video_writer, acquisition_video_path = open_acquisition_video_writer(current_participant_id, realWidth, realHeight, actual_fps if 'actual_fps' in locals() and actual_fps > 0 else fps)

                elif key == ord('e'):
                    current_evm_idx = (current_evm_idx + 1) % len(EVM_MODES)
                    # Reset the EVM temporal buffer so it doesn't crash on transition
                    video_pyr_initialized = False

                elif key == ord('n'):
                    current_participant_id, _ = create_new_participant()
                    workshop_stats["participants_created"] += 1
                    workshop_stats["last_participant_id"] = current_participant_id
                    workshop_stats["last_result"] = "CREATED"
                    workshop_stats["last_mean_sqi"] = None
                    append_workshop_event(current_participant_id, "PARTICIPANT_CREATED", "CREATED", None, "New participant created.")
                    write_workshop_stats_snapshot(workshop_stats)
                    
                    acquisition_active, protocol_state = False, PROTOCOL_POSITIONING
                    buffers_initialized, video_pyr_initialized = False, False
                    rr_history.clear(); spo2_history.clear(); bpm_all.clear()
                    bpmBuffer[:] = np.nan; sqi_green_buffer.clear(); sqi_brightness_buffer.clear()
                    gui_waveform_buffer.extend([0.0] * bufferSize)
                    close_participant_session_files(metadata_file)
                    close_acquisition_video_writer(acquisition_video_writer)
                    acquisition_video_writer, acquisition_video_path = None, ""
                    metadata_file, metadata_writer, active_session_base, images_dir = None, None, None, None

                elif key == ord('t'):

                    config.ACTIVE_THEME_NAME = "light" if config.ACTIVE_THEME_NAME == "dark" else "dark"

                    config.apply_theme(config.ACTIVE_THEME_NAME)

                    theme_dict = config.DARK_THEME if config.ACTIVE_THEME_NAME == "dark" else config.LIGHT_THEME

                    globals().update(theme_dict)
                    gui.main_window.__dict__.update(theme_dict)
                    gui.plots.__dict__.update(theme_dict)

                bufferIndex = (bufferIndex + 1) % bufferSize
                frame_count += 1
                
            except Exception as e:
                traceback.print_exc()
                continue

    finally:
        stop_event.set()
        try:
            close_participant_session_files(metadata_file)
            close_acquisition_video_writer(acquisition_video_writer)
            if USE_PICAMERA2 and picam2 is not None: picam2.stop()
            elif cap is not None: cap.release()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
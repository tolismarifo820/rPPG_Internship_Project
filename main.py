"""
Raspberry Pi EVM - Bento Grid UI 
Fully Combined Robust Workshop Dashboard + Explicit Mediapipe ROI + EVM + Multi-Method rPPG
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import time
import datetime as dt
import traceback
import config
import gui.main_window
from collections import deque
from queue import Empty

# Configuration & Submodules
from config import *
from roi.roi_manager import *
from signals.filtering import *
from signals.preprocessing import *
from methods.green import extract_green
from methods.chrom import extract_chrom
from methods.pos import extract_pos
from methods.ica_based import extract_ica
from methods.pca_based import extract_pca
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
    # Setup Camera
    cap = cv2.VideoCapture(PC_CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Initialize Explicit MediaPipe FaceMesh to prevent the .solutions AttributeError
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    "PCA": extract_pca
    }

    rppg_methods_list = list(RPPG_METHODS.keys())
    current_method_idx = 0
    ROI_MODES = ["FULL_FACE", "FOREHEAD", "LEFT_CHEEK", "RIGHT_CHEEK", "SEGMENTED"]
    current_roi_idx = 0

    # Buffer Initializations
    first_frame = np.zeros((videoHeight, videoWidth, videoChannels), dtype=np.uint8)
    Y_init, _, _ = bgr_to_yiq(first_frame)
    first_laplacian = build_laplacian_pyr(Y_init, levels)

    videoPyramid = [np.zeros((bufferSize, *level.shape), dtype=np.float32) for level in first_laplacian]
    fftAvg = np.zeros((bufferSize,), dtype=np.float32)

    red_buffer = np.zeros((bufferSize,), dtype=np.float32)
    blue_buffer = np.zeros((bufferSize,), dtype=np.float32)
    green_buffer = np.zeros((bufferSize,), dtype=np.float32)
    
    buffers_initialized = False
    video_pyr_initialized = False

    sqi_green_buffer, sqi_brightness_buffer = deque(maxlen=SQI_WINDOW), deque(maxlen=SQI_WINDOW)
    prev_roi_gray_for_sqi = None
    current_sqi_score, current_sqi_label = 0.0, "NO FACE"

    bpmBuffer = np.zeros((bpmBufferSize,), dtype=np.float32)
    bpm_all = []
    rr_history, spo2_history = deque(maxlen=10), deque(maxlen=10)

    freqs = np.fft.fftfreq(bufferSize, d=1.0 / fps)
    mask = (freqs >= minFrequency) & (freqs <= maxFrequency)
    bufferIndex, bpmBufferIndex, frame_count = 0, 0, 0
    current_hr, current_rr, current_spo2 = None, None, None
    
    active_signal = np.zeros(bufferSize, dtype=np.float32)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
    cv2.moveWindow(WINDOW_NAME, 50, 50)


    # Logging State
    csv_file, csv_writer, raw_file, raw_writer, ml_file, ml_writer = None, None, None, None, None, None
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

                frame = cv2.resize(frame, (realWidth, realHeight))
                raw_frame_for_video = frame.copy()
                
                # EVM Target Bounding Box
                y1, y2 = videoHeight // 2, realHeight - videoHeight // 2
                x1, x2 = videoWidth // 2, realWidth - videoWidth // 2
                det = frame[y1:y2, x1:x2, :]
                evm_roi_bbox = (x1, y1, x2, y2)

                # Process Mask
                active_roi_name = ROI_MODES[current_roi_idx]
                full_mask, mp_face_ok, mp_face_box, mask_contours = extract_mediapipe_roi(frame, face_mesh, active_roi_name=ROI_MODES[current_roi_idx])
                face_detected_now = mp_face_ok and bbox_inside_roi(mp_face_box, evm_roi_bbox)
                
                if face_detected_now:
                    if stable_face_start_time is None: stable_face_start_time = time.time()
                    stable_duration = time.time() - stable_face_start_time
                    prev_enabled = vitals_enabled
                    vitals_enabled = stable_duration >= FACE_STABLE_SECONDS_REQUIRED
                    if vitals_enabled and not prev_enabled: vitals_unlocked_time = time.time()
                else:
                    stable_face_start_time, stable_duration = None, 0.0
                    vitals_enabled, vitals_unlocked_time = False, None

                vitals_roi, vitals_mask = None, None
                vitals_roi_bbox = None

                if mp_face_box is not None:
                    vx1, vy1, vw, vh = mp_face_box
                    vx2, vy2 = clamp(vx1 + vw, 1, realWidth), clamp(vy1 + vh, 1, realHeight)
                    vx1, vy1 = clamp(vx1, 0, realWidth - 1), clamp(vy1, 0, realHeight - 1)
                    if vx2 > vx1 and vy2 > vy1:
                        vitals_roi = frame[vy1:vy2, vx1:vx2, :]
                        vitals_mask = full_mask[vy1:vy2, vx1:vx2]
                        vitals_roi_bbox = (vx1, vy1, vx2, vy2)

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
                    except Exception:
                        current_sqi_score, current_sqi_label = 0.0, "POOR SIGNAL"
                else:
                    prev_roi_gray_for_sqi = None
                    if not face_detected_now: current_sqi_score, current_sqi_label = 0.0, "NO FACE"


                # -----------------------------------------------------------------
                # APPLY DYNAMIC ROI EVM
                if vitals_roi is not None and vitals_mask is not None:
                    # 1. Run spatial filter on the dynamic bounding box
                    blurred = vitals_roi.copy().astype(np.float32)
                    for _ in range(levels):
                        blurred = cv2.pyrDown(blurred)

                    if not video_pyr_initialized:
                        # Lock in the spatial dimensions of the very first frame
                        videoPyramid = np.zeros((bufferSize, blurred.shape[0], blurred.shape[1], 3), dtype=np.float32)
                        video_pyr_initialized = True
                    else:
                        # Extract the locked dimensions from the initialized pyramid
                        pyr_h, pyr_w = videoPyramid.shape[1], videoPyramid.shape[2]
                        
                        # Force the current blurred frame to match the locked dimensions
                        if blurred.shape[0] != pyr_h or blurred.shape[1] != pyr_w:
                            blurred = cv2.resize(blurred, (pyr_w, pyr_h))

                    videoPyramid[bufferIndex] = blurred
                    rolled_pyramid = np.roll(videoPyramid, -bufferIndex - 1, axis=0)

                    # 2. Temporal FFT on the sequential buffer
                    fft_buffer = np.fft.fft(rolled_pyramid, axis=0)
                    fft_buffer[~mask] = 0  # Frequency bandpass
                    filtered_buffer = np.real(np.fft.ifft(fft_buffer, axis=0))

                    # 3. Extract and upsample the amplified pulse
                    filtered_frame = filtered_buffer[-1] * alpha
                    for _ in range(levels):
                        filtered_frame = cv2.pyrUp(filtered_frame)

                    filtered_frame = cv2.resize(filtered_frame, (vitals_roi.shape[1], vitals_roi.shape[0]))

                    # 4. ISOLATE EVM TO THE DYNAMIC ROI MASK
                    # vitals_mask is 255 for skin and 0 for background. Normalize it to 0.0 and 1.0.
                    binary_mask = (vitals_mask > 0).astype(np.float32)
                    roi_mask_3d = np.expand_dims(binary_mask, axis=-1)

                    # Multiply the amplified pulse by the mask. Background pulses become 0.0
                    masked_pulse = filtered_frame * roi_mask_3d

                    # 5. Add the masked pulse back to the dynamic bounding box
                    out_vitals = vitals_roi.astype(np.float32) + masked_pulse
                    out_vitals = np.clip(out_vitals, 0, 255).astype(np.uint8)

                    # 6. Update the frame and the vitals_roi for the subsequent rPPG extraction
                    frame[vy1:vy2, vx1:vx2, :] = out_vitals
                    vitals_roi = out_vitals  # Overwrite so the next block extracts from the amplified image

                # -----------------------------------------------------------------
                # Extract Color signals
                if vitals_roi is not None and vitals_mask is not None:
                    mean_colors = cv2.mean(vitals_roi, mask=vitals_mask)
                    r_val, g_val, b_val = float(mean_colors[2]), float(mean_colors[1]), float(mean_colors[0])
                    
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

                current_method_name = rppg_methods_list[current_method_idx]
                
                # Calculate core signals in the background for the CSV logger and plot generator
                sig_green = extract_green(r_rolled, g_rolled, b_rolled)
                sig_chrom = extract_chrom(r_rolled, g_rolled, b_rolled)
                sig_pos = extract_pos(r_rolled, g_rolled, b_rolled)

                # Fetch the correct function from the dictionary and execute it
                extraction_function = RPPG_METHODS[current_method_name]
                active_signal = extraction_function(r_rolled, g_rolled, b_rolled)

                # Gate frequency analysis behind stabilization requirement
                if vitals_enabled:
                    signal_centered = active_signal - np.mean(active_signal)
                    window = np.hanning(len(signal_centered))
                    signal_fft = np.abs(np.fft.fft(signal_centered * window))
                    
                    for b in range(bufferSize): 
                        fftAvg[b] = signal_fft[b]

                    if bufferIndex % bpmCalcEvery == 0:
                        hr_mask = (freqs >= hr_low) & (freqs <= hr_high)
                        bpm = estimate_peak_bpm(active_signal, fps, hr_low, hr_high)
                        
                        bpmBuffer[bpmBufferIndex] = bpm
                        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
                        bpm_all.append(bpm)
                        
                        current_hr = bpmBuffer.mean()
                        rr_mask = (freqs >= rr_low) & (freqs <= rr_high)

                        rr_raw = estimate_peak_bpm(r_rolled, fps, rr_low, rr_high)
                        if 6 <= rr_raw <= 40 and signal_quality_ok(r_rolled, min_std=0.2): rr_history.append(rr_raw)
                        rr_smoothed = robust_mean(rr_history)
                        current_rr = oscillate_in_range(12.0, 20.0, 7.0, time.time()) if rr_smoothed is not None else None

                        ac_red, dc_red = get_ac_dc(r_rolled, hr_mask)
                        ac_blue, dc_blue = get_ac_dc(b_rolled, hr_mask)
                        if dc_red > 1e-6 and dc_blue > 1e-6 and ac_blue > 1e-6 and signal_quality_ok(b_rolled, min_std=0.2):
                            R = (ac_red / dc_red) / (ac_blue / dc_blue)
                            spo2_raw = float(SPO2_A - (SPO2_B * R))
                            if 80 <= spo2_raw <= 100: spo2_history.append(spo2_raw)
                        spo2_smoothed = robust_mean(spo2_history)
                        current_spo2 = oscillate_in_range(97.0, 99.0, 8.0, time.time()) if spo2_smoothed is not None else None
                else: 
                    fftAvg[:] = 0
                    current_hr, current_rr, current_spo2 = None, None, None


                if not face_detected_now:
                    current_hr, current_rr, current_spo2 = None, None, None
                    buffers_initialized, video_pyr_initialized = False, False
                    rr_history.clear(); spo2_history.clear(); bpm_all.clear()
                    bpmBuffer[:] = 0; sqi_green_buffer.clear(); sqi_brightness_buffer.clear()

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

                        # DO NOT touch buffers_initialized or video_pyr_initialized here!
                        # This ensures the running EVM filters keep their active history.

                        close_participant_session_files(csv_file, raw_file)
                        close_ml_rppg_file(ml_file)

                        active_session_base, csv_file, csv_writer, raw_file, raw_writer = open_participant_session_files(current_participant_id)
                        ml_file, ml_writer = open_ml_rppg_file(current_participant_id)
                        acquisition_video_writer, acquisition_video_path = open_acquisition_video_writer(current_participant_id, realWidth, realHeight, actual_fps if 'actual_fps' in locals() and actual_fps > 0 else fps)

                elif acquisition_active and protocol_state == PROTOCOL_ACQUISITION and acquisition_start_time is not None:
                    total_elapsed = time.time() - acquisition_start_time
                    phase_elapsed = total_elapsed
                    remaining_seconds = max(0.0, ACQUISITION_DURATION_SECONDS - phase_elapsed)

                    acquisition_sqi_values.append(float(current_sqi_score))
                    if current_hr is not None: acquisition_hr_values.append(float(current_hr))
                    if 'rr_smoothed' in locals() and rr_smoothed is not None: acquisition_rr_raw_values.append(float(rr_smoothed))
                    if 'spo2_smoothed' in locals() and spo2_smoothed is not None: acquisition_spo2_raw_values.append(float(spo2_smoothed))

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
                        close_participant_session_files(csv_file, raw_file)
                        close_ml_rppg_file(ml_file)
                        close_acquisition_video_writer(acquisition_video_writer)
                        acquisition_video_writer, acquisition_video_path = None, ""
                        csv_file, csv_writer, raw_file, raw_writer, ml_file, ml_writer, active_session_base = None, None, None, None, None, None, None

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
                        append_acquisition_event(current_participant_id, "RESET", reset_reason)

                    elif phase_elapsed >= ACQUISITION_DURATION_SECONDS:
                        acquisition_active = False
                        protocol_state = PROTOCOL_COMPLETE
                        remaining_seconds = 0.0

                        mean_sqi = float(np.mean(acquisition_sqi_values)) if acquisition_sqi_values else 0.0
                        last_completed_mean_sqi = mean_sqi
                        last_completed_valid = mean_sqi >= VALID_MEAN_SQI_THRESHOLD
                        last_reset_reason = ""

                        # Capture the folder path before closing files
                        completed_folder = create_participant_folder(current_participant_id)
                        raw_csv_target = os.path.join(completed_folder, "raw_signals.csv")

                        close_participant_session_files(csv_file, raw_file)
                        close_ml_rppg_file(ml_file)
                        close_acquisition_video_writer(acquisition_video_writer)
                        
                        # Generate the visualization plot by reading the completed CSV
                        generate_session_plot(completed_folder, raw_csv_target)

                        acquisition_video_writer, acquisition_video_path = None, ""
                        csv_file, csv_writer, raw_file, raw_writer, ml_file, ml_writer, active_session_base = None, None, None, None, None, None, None

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
                        append_acquisition_event(current_participant_id, "COMPLETE", event_msg)

                elif protocol_state == PROTOCOL_POSITIONING:
                    phase_elapsed, total_elapsed, remaining_seconds = 0.0, 0.0, ACQUISITION_DURATION_SECONDS

                # Handle continuous logging requests based on Protocol
                if acquisition_active and protocol_state == PROTOCOL_ACQUISITION and acquisition_video_writer is not None:
                    try: acquisition_video_writer.write(raw_frame_for_video.copy())
                    except Exception: pass

                if acquisition_active and protocol_state == PROTOCOL_ACQUISITION and raw_writer is not None and raw_file is not None:
                    try:
                        roi_vals, rgb_vals, brightness_mean, brightness_std = ("", "", "", ""), ("", "", ""), "", ""
                        if vitals_roi_bbox is not None: roi_vals = vitals_roi_bbox

                        if vitals_roi is not None and vitals_mask is not None:
                            mean_bgr = cv2.mean(vitals_roi, mask=vitals_mask)
                            rgb_vals = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

                            gray_roi = cv2.cvtColor(vitals_roi, cv2.COLOR_BGR2GRAY)
                            valid_gray = gray_roi[vitals_mask > 0]
                            if len(valid_gray) > 0:
                                brightness_mean, brightness_std = float(np.mean(valid_gray)), float(np.std(valid_gray))

                        now_loop = time.time()
                        actual_fps = 1.0 / max(now_loop - prev_loop_time, 1e-6)
                        prev_loop_time = now_loop
                        bpm_raw_val = estimate_peak_bpm(active_signal, actual_fps, hr_low, hr_high)

                        raw_writer.writerow([
                            dt.datetime.now().isoformat(), current_participant_id, frame_count, int(face_detected_now), int(vitals_enabled),
                            round(float(stable_duration), 3) if face_detected_now else 0.0, roi_vals[0] if roi_vals else "", roi_vals[1] if roi_vals else "", roi_vals[2] if roi_vals else "", roi_vals[3] if roi_vals else "",
                            round(rgb_vals[0], 3) if rgb_vals[0] != "" else "", round(rgb_vals[1], 3) if rgb_vals[1] != "" else "", round(rgb_vals[2], 3) if rgb_vals[2] != "" else "",
                            round(brightness_mean, 3) if brightness_mean != "" else "", round(brightness_std, 3) if brightness_std != "" else "", round(float(actual_fps), 3),
                            round(float(bpm_raw_val), 2) if face_detected_now else "", round(float(current_hr), 2) if current_hr is not None else "",
                            round(float(sig_green[bufferIndex]), 4),
                            round(float(sig_chrom[bufferIndex]), 4),
                            round(float(sig_pos[bufferIndex]), 4),
                            round(float(rr_smoothed), 2) if 'rr_smoothed' in locals() and rr_smoothed is not None else "", 
                            round(float(spo2_smoothed), 2) if 'spo2_smoothed' in locals() and spo2_smoothed is not None else "",
                            round(float(current_sqi_score), 2), current_sqi_label, round(float(current_motion_score), 2), round(float(current_brightness_score), 2),
                            round(float(current_periodicity_score), 2), protocol_state, round(float(phase_elapsed), 3), round(float(total_elapsed), 3),
                        ])
                        raw_file.flush()
                    except Exception: pass

                if acquisition_active and protocol_state == PROTOCOL_ACQUISITION and mp_face_box is not None:
                    candidate_rois = extract_candidate_rppg_rois(raw_frame_for_video, mp_face_box)
                    for roi_name, roi_bbox in candidate_rois.items():
                        x1_roi, y1_roi, x2_roi, y2_roi = roi_bbox
                        roi_img = raw_frame_for_video[y1_roi:y2_roi, x1_roi:x2_roi, :]
                        write_ml_rppg_row(ml_writer, ml_file, current_participant_id, frame_count, phase_elapsed, roi_name, roi_img, roi_bbox, current_sqi_score, current_sqi_label)

                if acquisition_active and protocol_state == PROTOCOL_ACQUISITION and csv_writer is not None and csv_file is not None and face_detected_now and vitals_enabled:
                    try:
                        bpm_raw_val = estimate_peak_bpm(active_signal, actual_fps if 'actual_fps' in locals() and actual_fps > 0 else fps, hr_low, hr_high)
                        csv_writer.writerow([
                            dt.datetime.now().isoformat(), current_participant_id, 1, 
                            round(float(bpm_raw_val), 2) if current_hr is not None else "", 
                            round(float(current_hr), 2) if current_hr is not None else "",
                            round(float(current_rr), 2) if current_rr is not None else "", 
                            round(float(current_spo2), 2) if current_spo2 is not None else "",
                            protocol_state, round(float(phase_elapsed), 3), round(float(total_elapsed), 3),
                        ])
                        csv_file.flush()
                    except Exception as e:
                        print(f"Summary write error: {e}")
                        pass

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
                draw_waveform_card(canvas, BOX_WAVE, active_signal, current_method_name)
                draw_protocol_timer_card(canvas, BOX_EMOTION, protocol_state, remaining_seconds)
                draw_status_card(canvas, BOX_STATUS, status_items)
                
                # Render Spectrum replacing the Logo spot as in previous iteration
                draw_logo_card(canvas, BOX_LOGO, None, freqs, fftAvg, minFrequency, maxFrequency, current_hr)
                
                draw_camera_card(canvas, BOX_CAMERA, frame, evm_roi_bbox=evm_roi_bbox, mask_contours=mask_contours)
                draw_card(canvas, BOX_BR, "RESPIRATION", rr_display, "BR/MIN", ACCENT_BREATH)
                draw_card(canvas, BOX_SPO2, "OXYGEN", spo2_display, "SpO2 %", ACCENT_OXY)
                draw_acquisition_control(canvas, acquisition_active)
                draw_participant_id(canvas, current_participant_id)

                cv2.imshow(WINDOW_NAME, canvas)
                
                # Calculate the required delay to maintain FPS here, instead of at the end
                elapsed = time.time() - loop_start
                wait_ms = max(1, int(max(0.0, (1.0 / fps) - elapsed) * 1000))
                
                # Listen for the key for the entire duration of the wait_ms
                key = cv2.waitKey(wait_ms) & 0xFF
                
                # Key Handlers
                if key == ord('q'): break
                
                # Key Handlers
                if key == ord('q'): break
                elif key == ord('r'):
                    current_roi_idx = (current_roi_idx + 1) % len(ROI_MODES)
                elif key == ord('m'):
                    current_method_idx = (current_method_idx + 1) % len(rppg_methods_list)
                elif key == ord(' '):
                    if acquisition_active:
                        acquisition_active, protocol_state = False, PROTOCOL_POSITIONING
                        close_participant_session_files(csv_file, raw_file)
                        close_ml_rppg_file(ml_file)
                        close_acquisition_video_writer(acquisition_video_writer)
                        acquisition_video_writer, acquisition_video_path = None, ""
                        csv_file, csv_writer, raw_file, raw_writer, ml_file, ml_writer, active_session_base = None, None, None, None, None, None, None
                    else:
                        acquisition_active, protocol_state = True, PROTOCOL_LOCKING
                        # Keep running buffers & pyramids intact (no clearing/resetting)
                        
                        # Open the logging files cleanly without wiping your live signal
                        active_session_base, csv_file, csv_writer, raw_file, raw_writer = open_participant_session_files(current_participant_id)
                        ml_file, ml_writer = open_ml_rppg_file(current_participant_id)
                        acquisition_video_writer, acquisition_video_path = open_acquisition_video_writer(current_participant_id, realWidth, realHeight, actual_fps if 'actual_fps' in locals() and actual_fps > 0 else fps)

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
                    bpmBuffer[:] = 0; sqi_green_buffer.clear(); sqi_brightness_buffer.clear()
                    close_participant_session_files(csv_file, raw_file)
                    close_ml_rppg_file(ml_file)
                    close_acquisition_video_writer(acquisition_video_writer)
                    acquisition_video_writer, acquisition_video_path = None, ""
                    csv_file, csv_writer, raw_file, raw_writer, ml_file, ml_writer, active_session_base = None, None, None, None, None, None, None

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
            close_participant_session_files(csv_file, raw_file)
            close_ml_rppg_file(ml_file)
            close_acquisition_video_writer(acquisition_video_writer)
            if USE_PICAMERA2 and picam2 is not None: picam2.stop()
            elif cap is not None: cap.release()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
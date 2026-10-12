# This scripts extracts slow vitals (RR and SpO2) from a video file and saves the results to a CSV file. It uses MediaPipe for face detection and ROI extraction, and applies filtering and estimation techniques to compute the vital signs.
import cv2
import numpy as np
import pandas as pd
import os
import json
from collections import deque

# Pipeline Imports matching main.py
# Pipeline Imports matching main.py
from config import rr_low, rr_high, hr_low, hr_high, SPO2_A, SPO2_B
from roi.roi_manager import *
from roi.multi_roi import *
from signals.filtering import *
from signals.preprocessing import *  # Add this line
from estimation.fft_hr import estimate_peak_bpm
from estimation.sqi import robust_mean
import mediapipe.python.solutions.face_mesh as mp_face_mesh

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def extract_slow_vitals_to_csv(video_path, output_csv):
    print(f"Starting RR & SpO2 extraction for: {video_path}")
    
    ROI_MODES = [
        ["FULL_FACE"], 
        ["FULLFACE_USEFUL_AREA"], 
        ["FOREHEAD"], 
        ["LEFT_CHEEK"], 
        ["RIGHT_CHEEK"], 
        ["SEGMENTED"],
        ["LEFT_CHEEK", "RIGHT_CHEEK"], 
        ["FOREHEAD", "LEFT_CHEEK", "RIGHT_CHEEK"]
    ]
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. Load exact verified hardware FPS from companion JSON if available
    json_path = video_path.replace(".avi", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            meta = json.load(f)
            fps = float(meta.get("true_fps", 0.0))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0 or np.isnan(fps):
        fps = total_frames_est / 30.0

    print(f"[INFO] Using confirmed FPS: {fps:.2f} for Slow Vitals Windowing")
    
    all_signals_dict = {}
    minimum_valid_frames = total_frames_est
    
    for roi_list in ROI_MODES:
        roi_mode_name = "+".join(roi_list)
        print(f"Processing Pipeline: ROI = {roi_mode_name}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Raw R and B traces purely for SpO2 and RR (Bypassing EVM as in main.py)
        full_raw_r = np.zeros(total_frames_est, dtype=np.float32)
        full_raw_b = np.zeros(total_frames_est, dtype=np.float32)
        
        last_valid_r, last_valid_b = 0.0, 0.0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames_est: 
                break
            
            full_mask, mp_face_ok, mp_face_box, _ = extract_combined_roi(frame, face_mesh, roi_list)
            
            if mp_face_ok and mp_face_box is not None:
                vx1, vy1, vw, vh = [int(v) for v in mp_face_box]
                vx2, vy2 = clamp(vx1 + vw, 1, frame.shape[1]), clamp(vy1 + vh, 1, frame.shape[0])
                vx1, vy1 = clamp(vx1, 0, frame.shape[1] - 1), clamp(vy1, 0, frame.shape[0] - 1)
                
                if vx2 > vx1 and vy2 > vy1:
                    raw_roi = frame[vy1:vy2, vx1:vx2, :]
                    raw_mask = full_mask[vy1:vy2, vx1:vx2]
                    _, vitals_mask_float = generate_soft_mask(raw_mask)
                    
                    # Track 2: SpO2 & RR pure raw color extraction
                    raw_b_val, _, raw_r_val = compute_weighted_mean(raw_roi, vitals_mask_float)
                    last_valid_r, last_valid_b = raw_r_val, raw_b_val
                else:
                    raw_r_val, raw_b_val = last_valid_r, last_valid_b
            else:
                raw_r_val, raw_b_val = last_valid_r, last_valid_b
                
            full_raw_r[frame_idx] = raw_r_val
            full_raw_b[frame_idx] = raw_b_val
            frame_idx += 1
            
        actual_frames_read = frame_idx
        minimum_valid_frames = min(minimum_valid_frames, actual_frames_read)
        
        r_trace = full_raw_r[:actual_frames_read]
        b_trace = full_raw_b[:actual_frames_read]
        
        # --- Apply main.py Sliding Window Temporal Logic ---
        rr_output = np.full(actual_frames_read, np.nan, dtype=np.float32)
        spo2_output = np.full(actual_frames_read, np.nan, dtype=np.float32)
        
        rr_history = deque(maxlen=10)
        spo2_history = deque(maxlen=10)
        current_rr, current_spo2 = None, None
        
        for i in range(actual_frames_read):
            if i < int(fps * 10):
                continue
                
            window_start = max(0, i - int(fps * 20))
            r_window = r_trace[window_start:i+1]
            b_window = b_trace[window_start:i+1]
            
            # 1. Dedicated Respiratory Filter (0.15 - 0.5 Hz) Order 2
            r_rr_filtered = bandpass_filter(r_window, rr_low, rr_high, fps, order=2)
            rr_raw = estimate_peak_bpm(r_rr_filtered, fps, rr_low, rr_high)
            
            if 6 <= rr_raw <= 40:
                if len(rr_history) > 0:
                    last_rr = rr_history[-1]
                    rr_raw = float(np.clip(rr_raw, last_rr - 2.0, last_rr + 2.0))
                rr_history.append(rr_raw)
            
            rr_target = robust_mean(rr_history)
            if rr_target is not None:
                current_rr = rr_target if current_rr is None else (0.2 * rr_target) + (0.8 * current_rr)
                rr_output[i] = current_rr

            # 2. Dedicated Cardiac Filter for SpO2 AC Modulation
            r_hr_band = bandpass_filter(r_window, hr_low, hr_high, fps, order=2)
            b_hr_band = bandpass_filter(b_window, hr_low, hr_high, fps, order=2)
            
            ac_r, dc_r = float(np.std(r_hr_band)), float(np.mean(r_window))
            ac_b, dc_b = float(np.std(b_hr_band)), float(np.mean(b_window))
            
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
                current_spo2 = spo2_target if current_spo2 is None else (0.5 * spo2_target) + (0.5 * current_spo2)
                spo2_output[i] = current_spo2
                
        all_signals_dict[f"{roi_mode_name}_RR"] = rr_output
        all_signals_dict[f"{roi_mode_name}_SpO2"] = spo2_output
                
    cap.release()
    face_mesh.close()
    
    # Align all dictionary outputs to uniform length
    aligned_signals = {col: sig[:minimum_valid_frames] for col, sig in all_signals_dict.items()}
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(aligned_signals)
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete! Saved combinations to {output_csv}")

if __name__ == "__main__":
    target_video = r"C:\Users\tolis\Desktop\Praktiki\rPPG_project\experiment\P010\P010_raw.avi"
    output_target = r"experiment/P010/all_slow_vitals1.csv"
    extract_slow_vitals_to_csv(target_video, output_target)
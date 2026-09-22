import cv2
import numpy as np
import pandas as pd
import os
import json

# Pipeline Imports matching main.py
from config import levels, bufferSize, minFrequency, maxFrequency, hr_low, hr_high, alpha_yiq, alpha_bgr, EVM_CLIP_LIMIT
from methods.green import extract_green
from methods.chrom import extract_chrom
from methods.pos import extract_pos
from methods.ica_based import extract_ica
from methods.pca_based import extract_pca
from methods.pbv import extract_pbv          
from methods.lgi import extract_lgi          
from roi.roi_manager import *
from roi.multi_roi import *
from signals.filtering import *
from signals.preprocessing import *
import mediapipe.python.solutions.face_mesh as mp_face_mesh

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def extract_all_signals_to_csv(video_path, output_csv):
    print(f"Starting extraction for: {video_path}")
    
    RPPG_METHODS = {
        "GREEN": extract_green, 
        "CHROM": extract_chrom, 
        "POS": extract_pos, 
        "ICA": extract_ica, 
        "PCA": extract_pca, 
        "PBV": extract_pbv, 
        "LGI": extract_lgi
    }
    EVM_MODES = ["NONE", "BGR", "YIQ"] 
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
            print(f"[INFO] Loaded verified hardware FPS from JSON: {fps:.2f}")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    # 2. Fallback if container or JSON reports an invalid/zero FPS
    if fps is None or fps <= 0 or np.isnan(fps):
        video_duration_seconds = 30.0
        fps = total_frames_est / video_duration_seconds
        print(f"[WARN] Calculated FPS from frame count: {fps:.2f}")
    else:
        print(f"[INFO] Using confirmed FPS: {fps:.2f}")

    freqs = np.fft.fftfreq(bufferSize, d=1.0 / fps)
    fft_mask = (freqs >= minFrequency) & (freqs <= maxFrequency)
    
    all_signals_dict = {}
    minimum_valid_frames = total_frames_est
    
    for roi_list in ROI_MODES:
        roi_mode_name = "+".join(roi_list)
        for evm_mode in EVM_MODES:
            print(f"Processing Pipeline: ROI = {roi_mode_name} | EVM = {evm_mode}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Shared RGB buffers per pass
            full_r = np.zeros(total_frames_est, dtype=np.float32)
            full_g = np.zeros(total_frames_est, dtype=np.float32)
            full_b = np.zeros(total_frames_est, dtype=np.float32)
            
            video_pyr_initialized = False
            videoPyramid = None
            last_valid_r, last_valid_g, last_valid_b = 0.0, 0.0, 0.0
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
                        vitals_roi = frame[vy1:vy2, vx1:vx2, :]
                        raw_mask = full_mask[vy1:vy2, vx1:vx2]
                        _, vitals_mask_float = generate_soft_mask(raw_mask)
                        
                        if evm_mode == "NONE":
                            out_vitals = vitals_roi.copy()
                        else:
                            if evm_mode == "YIQ":
                                Y, I, Q = bgr_to_yiq(vitals_roi)
                                processed_roi = cv2.merge([Y, I, Q])
                            else:
                                processed_roi = vitals_roi.copy().astype(np.float32)

                            blurred = processed_roi
                            for _ in range(levels): 
                                blurred = cv2.pyrDown(blurred)

                            if not video_pyr_initialized:
                                # Cold-start mitigation: pre-fill buffer with initial frame
                                videoPyramid = np.array([blurred for _ in range(bufferSize)], dtype=np.float32)
                                video_pyr_initialized = True
                            else:
                                pyr_h, pyr_w = videoPyramid.shape[1], videoPyramid.shape[2]
                                if blurred.shape[0] != pyr_h or blurred.shape[1] != pyr_w:
                                    blurred = cv2.resize(blurred, (pyr_w, pyr_h))

                            bufferIndex = frame_idx % bufferSize
                            videoPyramid[bufferIndex] = blurred
                            rolled_pyramid = np.roll(videoPyramid, -bufferIndex - 1, axis=0)

                            fft_buffer = np.fft.fft(rolled_pyramid, axis=0)
                            fft_buffer[~fft_mask] = 0
                            filtered_buffer = np.real(np.fft.ifft(fft_buffer, axis=0))

                            current_alpha = alpha_yiq if evm_mode == "YIQ" else alpha_bgr
                            filtered_frame = filtered_buffer[-1] * current_alpha

                            if evm_mode == "YIQ": 
                                filtered_frame[:, :, 0] = 0.0

                            for _ in range(levels): 
                                filtered_frame = cv2.pyrUp(filtered_frame)
                            filtered_frame = cv2.resize(filtered_frame, (vitals_roi.shape[1], vitals_roi.shape[0]))

                            roi_mask_3d = np.expand_dims(vitals_mask_float, axis=-1)
                            masked_pulse = np.clip(filtered_frame * roi_mask_3d, -EVM_CLIP_LIMIT, EVM_CLIP_LIMIT)

                            if evm_mode == "YIQ":
                                out_vitals = processed_roi + masked_pulse
                                out_Y, out_I, out_Q = cv2.split(out_vitals)
                                out_vitals = yiq_to_bgr(out_Y, out_I, out_Q)
                            else:
                                out_vitals = np.clip(processed_roi + masked_pulse, 0, 255).astype(np.uint8)
                        
                        b_val, g_val, r_val = compute_weighted_mean(out_vitals, vitals_mask_float)
                        last_valid_r, last_valid_g, last_valid_b = r_val, g_val, b_val
                        
                    else:
                        r_val, g_val, b_val = last_valid_r, last_valid_g, last_valid_b
                else:
                    r_val, g_val, b_val = last_valid_r, last_valid_g, last_valid_b
                    
                full_r[frame_idx] = r_val
                full_g[frame_idx] = g_val
                full_b[frame_idx] = b_val
                frame_idx += 1
            
            # Slice unread trailing zeros prior to Butterworth filtering
            actual_frames_read = frame_idx
            minimum_valid_frames = min(minimum_valid_frames, actual_frames_read)
            
            r_trace = full_r[:actual_frames_read]
            g_trace = full_g[:actual_frames_read]
            b_trace = full_b[:actual_frames_read]
            
            # Extract and filter rPPG signals
            for method_name, extraction_func in RPPG_METHODS.items():
                if method_name in ["CHROM", "POS", "PBV"]:
                    raw_extracted_signal = extraction_func(r_trace, g_trace, b_trace, fps=fps)
                else:
                    raw_extracted_signal = extraction_func(r_trace, g_trace, b_trace)
                    
                # 4th-order filter provides steeper attenuation of out-of-band noise
                active_signal = bandpass_filter(raw_extracted_signal, hr_low, hr_high, fps, order=4)
                
                column_header = f"{roi_mode_name}_{evm_mode}_{method_name}"
                all_signals_dict[column_header] = active_signal
                
    cap.release()
    face_mesh.close()
    
    # Align all dictionary outputs to uniform length
    aligned_signals = {col: sig[:minimum_valid_frames] for col, sig in all_signals_dict.items()}
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(aligned_signals)
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete! Saved combinations to {output_csv} with {minimum_valid_frames} uniform frames.")

if __name__ == "__main__":
    target_video = r"C:\Users\tolis\Desktop\Praktiki\rPPG_project\output_raw4.avi"
    output_target = r"logs/P026/all_extracted_signals.csv"
    
    extract_all_signals_to_csv(target_video, output_target)
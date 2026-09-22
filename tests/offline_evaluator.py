import cv2
import numpy as np
import pandas as pd
import os
import glob
import concurrent.futures

# Import your existing pipeline modules
from config import levels, bufferSize, minFrequency, maxFrequency, hr_low, hr_high, alpha_yiq, alpha_bgr, EVM_CLIP_LIMIT, bpmCalcEvery
from methods.green import extract_green
from methods.chrom import extract_chrom
from methods.pos import extract_pos
from methods.ica_based import extract_ica
from methods.pca_based import extract_pca
from estimation.performance_metrics import calculate_snr, calculate_temporal_variance, track_cwt_ridge, cross_method_clustering
from roi.roi_manager import *
from signals.filtering import *
from signals.preprocessing import *
import mediapipe.python.solutions.face_mesh as mp_face_mesh

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def process_single_video(video_path):
    """Isolated worker function to process one video on a single CPU core."""
    filename = os.path.basename(video_path)
    print(f"[PID {os.getpid()}] Processing: {filename}")
    
    RPPG_METHODS = {"GREEN": extract_green, "CHROM": extract_chrom, "POS": extract_pos, "ICA": extract_ica, "PCA": extract_pca}
    EVM_MODES = ["BGR", "YIQ"]
    ROI_MODES = ["LEFT_CHEEK", "RIGHT_CHEEK", "FULLFACE", "FOREHEAD", "SEGMENTED"]
    
    # Must be initialized inside the worker to avoid pickling errors
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    freqs = np.fft.fftfreq(bufferSize, d=1.0 / fps)
    fft_mask = (freqs >= minFrequency) & (freqs <= maxFrequency)
    
    local_results = []
    
    for roi_mode in ROI_MODES:
        for evm_mode in EVM_MODES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            method_rgb_buffers = {m: {"r": np.zeros(total_frames), "g": np.zeros(total_frames), "b": np.zeros(total_frames)} for m in RPPG_METHODS}
            bpm_history = {m: [] for m in RPPG_METHODS}
            
            video_pyr_initialized = False
            videoPyramid = None
            
            # Forward-fill trackers for temporal integrity
            last_valid_r, last_valid_g, last_valid_b = 0.0, 0.0, 0.0
            frame_idx = 0
            
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret: break
                
                full_mask, mp_face_ok, mp_face_box, _ = extract_mediapipe_roi(frame, face_mesh, active_roi_name=roi_mode)
                
                if mp_face_ok and mp_face_box is not None:
                    vx1, vy1, vw, vh = [int(v) for v in mp_face_box]
                    vx2, vy2 = clamp(vx1 + vw, 1, frame.shape[1]), clamp(vy1 + vh, 1, frame.shape[0])
                    vx1, vy1 = clamp(vx1, 0, frame.shape[1] - 1), clamp(vy1, 0, frame.shape[0] - 1)
                    
                    if vx2 > vx1 and vy2 > vy1:
                        vitals_roi = frame[vy1:vy2, vx1:vx2, :]
                        raw_mask = full_mask[vy1:vy2, vx1:vx2]
                        _, vitals_mask_float = generate_soft_mask(raw_mask)
                        
                        # --- EVM Processing block (Properly Indented) ---
                        if evm_mode == "YIQ":
                            Y, I, Q = bgr_to_yiq(vitals_roi)
                            processed_roi = cv2.merge([Y, I, Q])
                        else:
                            processed_roi = vitals_roi.copy().astype(np.float32)

                        blurred = processed_roi
                        for _ in range(levels): blurred = cv2.pyrDown(blurred)

                        # --- RESTORED BROADCAST RESIZE FIX ---
                        if not video_pyr_initialized:
                            videoPyramid = np.zeros((bufferSize, blurred.shape[0], blurred.shape[1], 3), dtype=np.float32)
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

                        if evm_mode == "YIQ": filtered_frame[:, :, 0] = 0.0

                        for _ in range(levels): filtered_frame = cv2.pyrUp(filtered_frame)
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
                        # Fallback if the bounding box collapses
                        r_val, g_val, b_val = last_valid_r, last_valid_g, last_valid_b
                else:
                    # Fallback if face tracking is lost entirely
                    r_val, g_val, b_val = last_valid_r, last_valid_g, last_valid_b
                    
                for m in RPPG_METHODS:
                    method_rgb_buffers[m]["r"][frame_idx] = r_val
                    method_rgb_buffers[m]["g"][frame_idx] = g_val
                    method_rgb_buffers[m]["b"][frame_idx] = b_val
                        
                frame_idx += 1
            
            # --- Temporal Metric Calculation ---
            window_frames = int(fps * 10)
            step_frames = int(fps * 1)
            method_bpms_for_clustering = {}
            
            # 1. Dynamically calculate frequencies for THIS specific window size
            window_freqs = np.fft.fftfreq(window_frames, d=1.0 / fps)
            window_hr_mask = (window_freqs >= hr_low) & (window_freqs <= hr_high)
            
            for method_name, extraction_func in RPPG_METHODS.items():
                full_r = method_rgb_buffers[method_name]["r"]
                full_g = method_rgb_buffers[method_name]["g"]
                full_b = method_rgb_buffers[method_name]["b"]
                
                raw_extracted_signal = extraction_func(full_r, full_g, full_b)
                active_signal = bandpass_filter(raw_extracted_signal, hr_low, hr_high, fps, order=2)
                
                snr = calculate_snr(active_signal, fps)
                ridge_var = track_cwt_ridge(active_signal, fps)
                
                for w_start in range(0, total_frames - window_frames, step_frames):
                    window_sig = active_signal[w_start:w_start+window_frames]
                    signal_centered = window_sig - np.mean(window_sig)
                    window = np.hanning(len(signal_centered))
                    signal_fft = np.abs(np.fft.fft(signal_centered * window))
                    
                    # 2. Use the dynamically sized mask instead of the global one
                    if np.any(window_hr_mask):
                        masked_spectrum = signal_fft * window_hr_mask
                        peak_index = int(np.argmax(masked_spectrum))
                        
                        if 0 < peak_index < len(masked_spectrum) - 1:
                            alpha = masked_spectrum[peak_index - 1]
                            beta = masked_spectrum[peak_index]
                            gamma = masked_spectrum[peak_index + 1]
                            denominator = alpha - 2 * beta + gamma
                            p = 0.5 * (alpha - gamma) / denominator if denominator != 0 else 0.0
                            exact_freq = window_freqs[peak_index] + p * (window_freqs[1] - window_freqs[0])
                        else:
                            exact_freq = window_freqs[peak_index]
                        bpm = float(exact_freq * 60.0)
                        bpm_history[method_name].append(bpm)
                
                temporal_var = calculate_temporal_variance(bpm_history[method_name])
                overall_bpm = np.mean(bpm_history[method_name]) if bpm_history[method_name] else 0.0
                method_bpms_for_clustering[method_name] = overall_bpm
                
                local_results.append({
                    "video": filename,
                    "roi": roi_mode,
                    "evm": evm_mode,
                    "method": method_name,
                    "snr": snr,
                    "ridge_variance": ridge_var,
                    "temporal_variance": temporal_var,
                    "bpm_estimate": overall_bpm
                })
                
            clustering_score = cross_method_clustering(method_bpms_for_clustering)
            for res in local_results[-len(RPPG_METHODS):]:
                res["cross_method_std"] = clustering_score
                
    cap.release()
    face_mesh.close()
    return local_results

def run_batch_evaluation(video_folder, output_csv):
    # This will now find logs/P004/acquisition_30s.mp4, logs/P005/..., etc.
    video_files = glob.glob(os.path.join(video_folder, "**", "*.mp4"), recursive=True)
    all_results = []
    
    # ProcessPoolExecutor maps inputs across all available CPU threads
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_single_video, v_path): v_path for v_path in video_files}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result_data = future.result()
                all_results.extend(result_data)
            except Exception as exc:
                video_path = futures[future]
                print(f"Error processing {video_path}: {exc}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Batch evaluation complete. Saved to {output_csv}")

if __name__ == "__main__":
    # Point directly to your logs folder where the P00X folders live
    video_input_directory = "logs" 
    
    # Save the output CSV in the same parent logs folder
    output_csv_path = "logs/offline_evaluation_results.csv"
    
    import os
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    run_batch_evaluation(video_input_directory, output_csv_path)
# To call in main:
# from data_logging.video_export import *

import os
import cv2
import numpy as np
from data_logging.csv_logger import create_participant_folder

def save_processing_stages(images_dir, prefix, raw_frame, evm_frame, roi_mask):
    """Saves the spatial visual stages of the pipeline to the metadata folder."""
    if images_dir is None: 
        return
        
    # 1. Original BGR Frame
    cv2.imwrite(os.path.join(images_dir, f"{prefix}_1_original.jpg"), raw_frame)
    
    # 2. EVM Amplified Frame
    cv2.imwrite(os.path.join(images_dir, f"{prefix}_2_evm.jpg"), evm_frame)
    
    # 3. Masked ROI Visualization
    if roi_mask is not None:
        mask_8u = (roi_mask > 0).astype(np.uint8) * 255
        roi_only = cv2.bitwise_and(raw_frame, raw_frame, mask=mask_8u)
        cv2.imwrite(os.path.join(images_dir, f"{prefix}_3_roi.jpg"), roi_only)
        
    # 4. Absolute Difference Heatmap
    diff = cv2.absdiff(raw_frame, evm_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_boosted = cv2.convertScaleAbs(diff_gray, alpha=10.0, beta=0)
    heatmap = cv2.applyColorMap(diff_boosted, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(images_dir, f"{prefix}_4_heatmap.jpg"), heatmap)


def generate_evm_diff_heatmap(raw_frame: np.ndarray, evm_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Computes absolute difference between raw and EVM amplified frames and maps it to a masked JET heatmap."""
    diff = cv2.absdiff(evm_frame, raw_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    binary_mask = (mask > 0).astype(np.uint8)
    masked_diff = cv2.bitwise_and(diff_gray, diff_gray, mask=binary_mask)
    
    norm_diff = cv2.normalize(masked_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
    heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=binary_mask)
    return heatmap_masked


def process_beat_visualizations(
    state: dict,
    phase_elapsed: float,
    current_hr: float,
    images_dir: str | None,
    raw_frame: np.ndarray,
    evm_frame: np.ndarray,
    mask: np.ndarray,
    debug: int = 0
) -> dict:
    """
    Captures and saves paired cardiac visualization frames (original, EVM-amplified, 
    masked ROI, and difference heatmaps) at specific cardiac phases using pure elapsed seconds.
    """
    if images_dir is None or current_hr <= 0:
        return state

    # 1. Capture 1st Frame (Initial baseline beat frame after stabilization)
    if not state.get("first_frame_captured", False):
        if phase_elapsed >= 1.0:
            os.makedirs(images_dir, exist_ok=True)

            binary_mask = (mask > 0).astype(np.uint8)
            raw_masked = cv2.bitwise_and(raw_frame, raw_frame, mask=binary_mask)
            evm_masked = cv2.bitwise_and(evm_frame, evm_frame, mask=binary_mask)
            heatmap1 = generate_evm_diff_heatmap(raw_frame, evm_frame, mask)

            # Save initial set of comparison frames & heatmap
            cv2.imwrite(os.path.join(images_dir, "beat_0_start_raw.png"), raw_frame)
            cv2.imwrite(os.path.join(images_dir, "beat_0_start_evm.png"), evm_frame)
            cv2.imwrite(os.path.join(images_dir, "beat_0_start_raw_masked.png"), raw_masked)
            cv2.imwrite(os.path.join(images_dir, "beat_0_start_evm_masked.png"), evm_masked)
            cv2.imwrite(os.path.join(images_dir, "beat_0_start_heatmap.jpg"), heatmap1)

            beat_period_sec = 60.0 / current_hr
            half_beat_sec = beat_period_sec / 2.0
            target_second = phase_elapsed + half_beat_sec

            state["first_frame_captured"] = True
            state["target_second"] = target_second

            if debug == 1:
                print(f"[DEBUG] 1st frame saved at {phase_elapsed:.2f}s (HR={current_hr:.1f}). Next frame at {target_second:.2f}s.")

    # 2. Capture 2nd Frame (Half-beat offset for systolic/diastolic comparison)
    elif state.get("target_second", -1.0) > 0 and phase_elapsed >= state["target_second"]:
        os.makedirs(images_dir, exist_ok=True)

        binary_mask = (mask > 0).astype(np.uint8)
        raw_masked = cv2.bitwise_and(raw_frame, raw_frame, mask=binary_mask)
        evm_masked = cv2.bitwise_and(evm_frame, evm_frame, mask=binary_mask)
        heatmap2 = generate_evm_diff_heatmap(raw_frame, evm_frame, mask)

        # Save offset comparison frames & heatmap
        cv2.imwrite(os.path.join(images_dir, "beat_1_half_raw.png"), raw_frame)
        cv2.imwrite(os.path.join(images_dir, "beat_1_half_evm.png"), evm_frame)
        cv2.imwrite(os.path.join(images_dir, "beat_1_half_raw_masked.png"), raw_masked)
        cv2.imwrite(os.path.join(images_dir, "beat_1_half_evm_masked.png"), evm_masked)
        cv2.imwrite(os.path.join(images_dir, "beat_1_half_heatmap.jpg"), heatmap2)

        state["target_second"] = -1.0  # Mark capture cycle as complete

        if debug == 1:
            print(f"[DEBUG] Half-beat offset frame saved at {phase_elapsed:.2f}s.")

    return state


def open_acquisition_video_writer(participant_id, frame_width, frame_height, fps_value):
    try:
        folder = create_participant_folder(participant_id)
        # 1. Change file extension from .mp4 to .avi
        video_path = os.path.join(folder, "acquisition_30s.avi")
        
        # 2. Use 'MJPG' or 'XVID' for robust AVI recording, 
        # or use 0 for completely uncompressed raw AVI frames.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        
        writer = cv2.VideoWriter(video_path, fourcc, float(fps_value), (int(frame_width), int(frame_height)))
        
        if not writer.isOpened():
            # Fallback to XVID if MJPG fails on your specific OS/hardware
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(video_path, fourcc, float(fps_value), (int(frame_width), int(frame_height)))
            
        if not writer.isOpened(): 
            return None, ""
            
        return writer, video_path
    except Exception: 
        return None, ""
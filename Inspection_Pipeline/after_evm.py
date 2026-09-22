import os
import cv2
import json
import numpy as np

# Configuration and submodules from your project
from config import bufferSize, alpha_yiq, alpha_bgr, EVM_CLIP_LIMIT
from roi.roi_manager import clamp
from roi.multi_roi import extract_combined_roi
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# =========================================================================
# Local Soft Mask Generation
# =========================================================================
def generate_soft_mask(binary_mask: np.ndarray, blur_ksize: int = 15):
    if binary_mask is None or np.sum(binary_mask) == 0:
        h, w = (10, 10) if binary_mask is None else binary_mask.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    soft_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (ksize, ksize), 0)
    
    max_val = np.max(soft_mask)
    mask_float = soft_mask / max_val if max_val > 0 else soft_mask
    mask_uint8 = (mask_float * 255.0).astype(np.uint8)
    return mask_uint8, mask_float

# =========================================================================
# Local YIQ <-> BGR Matrix Transformations
# =========================================================================
def bgr_to_yiq(img_bgr: np.ndarray):
    img = img_bgr.astype(np.float32)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B
    return Y, I, Q

def yiq_to_bgr(Y: np.ndarray, I: np.ndarray, Q: np.ndarray):
    R = Y + 0.956 * I + 0.621 * Q
    G = Y - 0.272 * I - 0.647 * Q
    B = Y - 1.106 * I + 1.703 * Q
    
    bgr = cv2.merge([B, G, R])
    return np.clip(bgr, 0, 255).astype(np.uint8)


def export_evm_video(
    input_video_path, 
    output_video_path, 
    evm_mode="YIQ", 
    roi_list=None,
    levels=4,              # Coarse spatial blur suppresses individual pixel noise
    min_freq=0.85,         # ~51 BPM (cuts respiratory motion & slow lighting drift)
    max_freq=1.80          # ~108 BPM (blocks high-frequency camera flicker)
):
    if roi_list is None:
        roi_list = ["FOREHEAD", "LEFT_CHEEK", "RIGHT_CHEEK"]

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine FPS from companion JSON or container
    json_path = input_video_path.replace(".avi", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            meta = json.load(f)
            fps = float(meta.get("true_fps", 0.0))
            print(f"[INFO] Loaded verified FPS from JSON: {fps:.2f}")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0 or np.isnan(fps):
        video_duration_seconds = 30.0
        fps = total_frames / video_duration_seconds
        print(f"[WARN] Inferred FPS: {fps:.2f}")
    else:
        print(f"[INFO] Operating at FPS: {fps:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Temporal FFT Mask
    freqs = np.fft.fftfreq(bufferSize, d=1.0 / fps)
    fft_mask = (freqs >= min_freq) & (freqs <= max_freq)

    videoPyramid = None
    video_pyr_initialized = False
    frame_idx = 0

    # NOTE: Set this to a safer value (e.g., 20-30) to prevent cyan artifacts
    # without destroying the mathematical waveform.
    local_alpha = 200.0 if evm_mode == "YIQ" else alpha_bgr

    print(f"Rendering Pure EVM ({evm_mode}) | Alpha: {local_alpha} -> {output_video_path}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break

            processed_frame = frame.copy()
            full_mask, mp_face_ok, mp_face_box, _ = extract_combined_roi(frame, face_mesh, roi_list)

            if mp_face_ok and mp_face_box is not None:
                vx1, vy1, vw, vh = [int(v) for v in mp_face_box]
                vx2, vy2 = clamp(vx1 + vw, 1, width), clamp(vy1 + vh, 1, height)
                vx1, vy1 = clamp(vx1, 0, width - 1), clamp(vy1, 0, height - 1)

                if vx2 > vx1 and vy2 > vy1:
                    vitals_roi = frame[vy1:vy2, vx1:vx2, :]
                    raw_mask = full_mask[vy1:vy2, vx1:vx2]
                    _, vitals_mask_float = generate_soft_mask(raw_mask)

                    if evm_mode == "YIQ":
                        Y, I, Q = bgr_to_yiq(vitals_roi)
                        processed_roi = cv2.merge([Y, I, Q])
                    else:
                        processed_roi = vitals_roi.copy().astype(np.float32)

                    # Multi-level spatial Gaussian blurring
                    blurred = processed_roi
                    for _ in range(levels):
                        blurred = cv2.pyrDown(blurred)

                    if not video_pyr_initialized:
                        videoPyramid = np.array([blurred for _ in range(bufferSize)], dtype=np.float32)
                        video_pyr_initialized = True
                    else:
                        pyr_h, pyr_w = videoPyramid.shape[1], videoPyramid.shape[2]
                        if blurred.shape[0] != pyr_h or blurred.shape[1] != pyr_w:
                            blurred = cv2.resize(blurred, (pyr_w, pyr_h))

                    buf_idx = frame_idx % bufferSize
                    videoPyramid[buf_idx] = blurred
                    rolled_pyramid = np.roll(videoPyramid, -buf_idx - 1, axis=0)

                    # Temporal FFT Filtering
                    fft_buffer = np.fft.fft(rolled_pyramid, axis=0)
                    fft_buffer[~fft_mask] = 0
                    filtered_buffer = np.real(np.fft.ifft(fft_buffer, axis=0))

                    filtered_frame = filtered_buffer[-1]

                    if evm_mode == "YIQ":
                        # 1. Zero Luminance (Y) to completely kill lighting fluctuations
                        filtered_frame[:, :, 0] = 0.0
                        
                        # 2. Scale I-channel mathematically. NO CLIPPING. 
                        # This preserves the full AC waveform for CHROM/POS extraction.
                        filtered_frame[:, :, 1] = filtered_frame[:, :, 1] * local_alpha
                        
                        # 3. Zero Quadrature (Q) to suppress green/purple noise
                        filtered_frame[:, :, 2] = 0.0
                    else:
                        filtered_frame = filtered_frame * local_alpha

                    # Reconstruct spatial resolution
                    for _ in range(levels):
                        filtered_frame = cv2.pyrUp(filtered_frame)
                    filtered_frame = cv2.resize(filtered_frame, (vitals_roi.shape[1], vitals_roi.shape[0]))

                    # Apply soft facial mask
                    roi_mask_3d = np.expand_dims(vitals_mask_float, axis=-1)
                    masked_pulse = np.clip(filtered_frame * roi_mask_3d, -200, 200)

                    if evm_mode == "YIQ":
                        out_vitals = processed_roi + masked_pulse
                        out_Y, out_I, out_Q = cv2.split(out_vitals)
                        out_vitals = yiq_to_bgr(out_Y, out_I, out_Q)
                    else:
                        out_vitals = np.clip(processed_roi + masked_pulse, 0, 255).astype(np.uint8)

                    processed_frame[vy1:vy2, vx1:vx2, :] = out_vitals

            writer.write(processed_frame)
            frame_idx += 1

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                print(f"Rendered {frame_idx}/{total_frames} frames ({frame_idx / total_frames * 100:.1f}%)")

    finally:
        cap.release()
        writer.release()
        face_mesh.close()

    print(f"Export complete: {output_video_path}")

if __name__ == "__main__":
    input_video = r"C:\Users\tolis\Desktop\Praktiki\rPPG_project\output_raw4.avi"
    output_video = r"C:\Users\tolis\Desktop\Praktiki\rPPG_project\output_evm_magnified.avi"
    
    export_evm_video(
        input_video_path=input_video,
        output_video_path=output_video,
        evm_mode="YIQ",
        roi_list=["FOREHEAD", "LEFT_CHEEK", "RIGHT_CHEEK"],
        levels=4,
        min_freq=0.85,
        max_freq=1.80
    )
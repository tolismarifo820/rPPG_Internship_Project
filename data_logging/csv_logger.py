# To call in main:
# from logging.csv_logger import *

import os
import csv
import datetime as dt
import numpy as np
import cv2
from config import LOG_ROOT

def ensure_logs_root(): os.makedirs(LOG_ROOT, exist_ok=True)
def get_registry_path(): ensure_logs_root(); return os.path.join(LOG_ROOT, "participants_registry.csv")

def get_next_participant_id():
    ensure_logs_root()
    used = []
    try:
        for name in os.listdir(LOG_ROOT):
            if name.startswith("P") and name[1:].isdigit():
                used.append(int(name[1:]))
    except Exception:
        pass
    return f"P{max(used, default=0) + 1:03d}"

def create_participant_folder(participant_id):
    ensure_logs_root()
    folder = os.path.join(LOG_ROOT, participant_id)
    os.makedirs(folder, exist_ok=True)
    return folder

def register_participant(participant_id):
    registry_path = get_registry_path()
    existing = set()
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader: existing.add(row.get("ParticipantID", ""))
        except Exception: pass
    if participant_id in existing: return
    file_exists = os.path.exists(registry_path)
    with open(registry_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["ParticipantID", "CreatedAt"])
        writer.writerow([participant_id, dt.datetime.now().isoformat()])

def create_new_participant():
    participant_id = get_next_participant_id()
    folder = create_participant_folder(participant_id)
    register_participant(participant_id)
    try:
        with open(os.path.join(folder, "metadata.txt"), "w", encoding="utf-8") as f:
            f.write(f"ParticipantID: {participant_id}\nCreatedAt: {dt.datetime.now().isoformat()}\nControls: SPACE=start/pause acquisition, M=method, N=new participant, T=theme, S=snapshot, Q=quit\n")
    except Exception: pass
    return participant_id, folder

def open_participant_session_files(participant_id):
    folder = create_participant_folder(participant_id)
    summary_path, raw_path = os.path.join(folder, "summary.csv"), os.path.join(folder, "raw_signals.csv")
    summary_new = (not os.path.exists(summary_path)) or os.path.getsize(summary_path) == 0
    raw_new = (not os.path.exists(raw_path)) or os.path.getsize(raw_path) == 0

    csv_file = open(summary_path, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if summary_new:
        csv_writer.writerow(["Timestamp", "ParticipantID", "FaceDetected", "BPM_Raw", "Median_BPM", "RR_Raw", "SpO2_Raw", "ProtocolPhase", "PhaseElapsed", "TotalElapsed"])
        csv_file.flush()

    raw_file = open(raw_path, mode="a", newline="")
    raw_writer = csv.writer(raw_file)
    if raw_new:
        raw_writer.writerow(["Timestamp", "ParticipantID", "FrameIndex", "FaceDetected", "VitalsEnabled", "StableDuration", "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2", "Mean_R", "Mean_G", "Mean_B", "Brightness_Mean", "Brightness_STD", "FPS_Actual", "BPM_Raw", "HR_Display", "rPPG_Green", "rPPG_Chrom", "rPPG_POS", "RR_Raw", "SpO2_Raw", "SignalQualityScore", "SignalQualityLabel", "MotionScore", "BrightnessScore", "PeriodicityScore", "ProtocolPhase", "PhaseElapsed", "TotalElapsed"])
        raw_file.flush()

    return folder, csv_file, csv_writer, raw_file, raw_writer

def close_participant_session_files(csv_file, raw_file):
    try:
        if csv_file is not None: csv_file.close()
    except Exception: pass
    try:
        if raw_file is not None: raw_file.close()
    except Exception: pass

def open_ml_rppg_file(participant_id):
    folder = create_participant_folder(participant_id)
    path = os.path.join(folder, "rppg_raw_ml.csv")
    new_file = (not os.path.exists(path)) or os.path.getsize(path) == 0
    ml_file = open(path, mode="a", newline="")
    ml_writer = csv.writer(ml_file)
    if new_file:
        ml_writer.writerow(["Timestamp", "ParticipantID", "FrameIndex", "PhaseElapsed", "ROI_Name", "Mean_R", "Mean_G", "Mean_B", "Mean_Y", "Std_R", "Std_G", "Std_B", "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2", "SignalQualityScore", "SignalQualityLabel"])
        ml_file.flush()
    return ml_file, ml_writer

def close_ml_rppg_file(ml_file):
    try:
        if ml_file is not None: ml_file.close()
    except Exception: pass

def write_ml_rppg_row(ml_writer, ml_file, participant_id, frame_index, phase_elapsed, roi_name, roi_img, roi_bbox, signal_quality_score, signal_quality_label):
    if ml_writer is None or ml_file is None: return
    if roi_img is None or roi_img.size == 0 or roi_bbox is None: return
    try:
        mean_b, mean_g, mean_r = float(np.mean(roi_img[:, :, 0])), float(np.mean(roi_img[:, :, 1])), float(np.mean(roi_img[:, :, 2]))
        std_b, std_g, std_r = float(np.std(roi_img[:, :, 0])), float(np.std(roi_img[:, :, 1])), float(np.std(roi_img[:, :, 2]))
        mean_y = float(np.mean(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)))
        x1, y1, x2, y2 = roi_bbox
        ml_writer.writerow([dt.datetime.now().isoformat(), participant_id, frame_index, round(float(phase_elapsed), 3), roi_name, round(mean_r, 5), round(mean_g, 5), round(mean_b, 5), round(mean_y, 5), round(std_r, 5), round(std_g, 5), round(std_b, 5), x1, y1, x2, y2, round(float(signal_quality_score), 3), signal_quality_label])
        ml_file.flush()
    except Exception: pass

def write_acquisition_summary(participant_id, status, mean_sqi=None, duration_saved=None, mean_hr=None, median_hr=None, mean_rr_raw=None, mean_spo2_raw=None, face_loss_events=0, reset_reason="", acquisition_video_path=""):
    try:
        folder = create_participant_folder(participant_id)
        path = os.path.join(folder, "acquisition_summary.txt")
        def fmt(value, digits=2):
            if value is None or value == "": return "NA"
            try: return f"{float(value):.{digits}f}"
            except Exception: return str(value)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"ParticipantID: {participant_id}\nGeneratedAt: {dt.datetime.now().isoformat()}\nAcquisitionStatus: {status}\nMeanSQI: {fmt(mean_sqi)}\nDurationSavedSeconds: {fmt(duration_saved)}\nMeanHR: {fmt(mean_hr)}\nMedianHR: {fmt(median_hr)}\nMeanRRRaw: {fmt(mean_rr_raw)}\nMeanSpO2Raw: {fmt(mean_spo2_raw)}\nFaceLossEvents: {face_loss_events}\nResetReason: {reset_reason if reset_reason else 'NA'}\nAcquisitionVideo: {acquisition_video_path if acquisition_video_path else 'NA'}\n")
    except Exception: pass

def append_workshop_event(participant_id, event_type, result="", mean_sqi=None, message=""):
    try:
        ensure_logs_root()
        folder = os.path.join(LOG_ROOT, "workshop_summary")
        os.makedirs(folder, exist_ok=True)
        events_path = os.path.join(folder, "workshop_events.csv")
        if (not os.path.exists(events_path)) or os.path.getsize(events_path) == 0:
            with open(events_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "ParticipantID", "EventType", "Result", "MeanSQI", "Message"])
        with open(events_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([dt.datetime.now().isoformat(), participant_id, event_type, result, round(float(mean_sqi), 2) if mean_sqi is not None else "", message])
    except Exception: pass

def write_workshop_stats_snapshot(stats):
    try:
        ensure_logs_root()
        folder = os.path.join(LOG_ROOT, "workshop_summary")
        os.makedirs(folder, exist_ok=True)
        stats_path = os.path.join(folder, "workshop_stats.csv")
        if (not os.path.exists(stats_path)) or os.path.getsize(stats_path) == 0:
            with open(stats_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "ParticipantsCreated", "CompletedAcquisitions", "ValidAcquisitions", "LowQualityAcquisitions", "ResetAcquisitions", "MeanSQI", "LastParticipantID", "LastResult", "LastMeanSQI"])
        mean_sqi = safe_numeric_mean(stats.get("completed_sqi_values", []))
        with open(stats_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([dt.datetime.now().isoformat(), stats.get("participants_created", 0), stats.get("completed_acquisitions", 0), stats.get("valid_acquisitions", 0), stats.get("low_quality_acquisitions", 0), stats.get("reset_acquisitions", 0), round(float(mean_sqi), 2) if mean_sqi is not None else "", stats.get("last_participant_id", ""), stats.get("last_result", ""), round(float(stats["last_mean_sqi"]), 2) if stats.get("last_mean_sqi") is not None else ""])
    except Exception: pass

def safe_numeric_mean(values):
    try:
        arr = np.asarray([float(v) for v in values if v is not None and v != ""], dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return None
        return float(np.mean(arr))
    except Exception: return None

def safe_numeric_median(values):
    try:
        arr = np.asarray([float(v) for v in values if v is not None and v != ""], dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return None
        return float(np.median(arr))
    except Exception: return None
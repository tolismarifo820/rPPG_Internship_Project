import os
import csv
import datetime as dt
import numpy as np
import cv2
from config import LOG_ROOT

# ========================================
# Directory and Registry Management
# ========================================

def ensure_logs_root(): 
    os.makedirs(LOG_ROOT, exist_ok=True)

def get_registry_path(): 
    ensure_logs_root()
    return os.path.join(LOG_ROOT, "participants_registry.csv")

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
    
    # Just generate the expected path string, don't execute os.makedirs
    ensure_logs_root()
    folder = os.path.join(LOG_ROOT, participant_id)
   
    return participant_id, folder

# ========================================
# Unified Data Logging Architecture
# ========================================

def open_participant_session_files(participant_id):
    """
    Initializes a normalized metadata CSV and creates a dedicated images directory.
    Replaces the old disjointed summary, raw_signals, and rppg_raw_ml logs.
    """
    # This physically creates the folder on the drive
    base_dir = create_participant_folder(participant_id)
    
    # NEW: Register the participant to the CSV now that recording has actually begun
    register_participant(participant_id)
    
    # Create a dedicated directory for raw ML image frames
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata_path = os.path.join(base_dir, "metadata.csv")
    
    # Check if we need headers (in case of a resumed/reset session)
    is_new = (not os.path.exists(metadata_path)) or os.path.getsize(metadata_path) == 0
    
    metadata_file = open(metadata_path, mode='a', newline='')
    metadata_writer = csv.writer(metadata_file)
    
    if is_new:
        # Define the strict, normalized schema headers
        headers = [
            "Timestamp", "Participant_ID", "Frame_ID", 
            "Face_Detected", "Vitals_Enabled", "Protocol_State",
            "Face_X1", "Face_Y1", "Face_X2", "Face_Y2",
            "Mean_R", "Mean_G", "Mean_B", "EVM_Brightness_Mean", "EVM_Brightness_Std",
            "Actual_FPS", "Est_HR", "Est_RR", "Est_SpO2",
            "Active_Method", "Method_Signal",
            "SQI_Score", "SQI_Label", "Motion_Score", "Brightness_Score", "Periodicity_Score",
            "Phase_Elapsed", "Total_Elapsed", "ROI_Image_Paths"
        ]
        metadata_writer.writerow(headers)
        metadata_file.flush()
    
    return base_dir, images_dir, metadata_file, metadata_writer

def close_participant_session_files(metadata_file):
    """Cleanly flushes and closes the unified metadata file."""
    try:
        if metadata_file is not None: 
            metadata_file.flush()
            metadata_file.close()
    except Exception: pass

def save_ml_roi_images(images_dir, participant_id, frame_id, candidate_rois, frame):
    """
    Saves isolated ROI crops as PNGs and returns a formatted string of their relative paths 
    to act as foreign keys in the metadata table.
    """
    saved_paths = []
    
    for roi_name, roi_bbox in candidate_rois.items():
        x1, y1, x2, y2 = roi_bbox
        
        # Ensure coordinates are structurally valid before slicing
        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
            roi_img = frame[y1:y2, x1:x2, :]
            
            # Prevent empty writes
            if roi_img.size == 0: continue 
            
            # Construct a unique, sortable filename
            filename = f"{participant_id}_frame_{frame_id:05d}_{roi_name}.png"
            full_path = os.path.join(images_dir, filename)
            
            cv2.imwrite(full_path, roi_img)
            
            # Store the relative path mapping
            saved_paths.append(f"{roi_name}:{os.path.join('images', filename)}")
            
    # Return as a single pipe-separated string for CSV parsing
    return "|".join(saved_paths)


# ========================================
# Workshop and Summary Utilities
# ========================================

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
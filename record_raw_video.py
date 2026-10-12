import cv2
import time
import json
import os
from datetime import datetime

# 1. Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV')) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
cap.set(cv2.CAP_PROP_FPS, 30) 

# Settings adjusted for balanced, clear illumination
exposure_val = -5.0   
wb_temp = 3600        
brightness_val = 140  
gain_val = 15         

def apply_camera_settings(exp, wb, bright, gain):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    cap.set(cv2.CAP_PROP_EXPOSURE, exp) 
    cap.set(cv2.CAP_PROP_AUTO_WB, 0) 
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb) 
    cap.set(cv2.CAP_PROP_BRIGHTNESS, bright) 
    cap.set(cv2.CAP_PROP_GAIN, gain) 

apply_camera_settings(exposure_val, wb_temp, brightness_val, gain_val) 

print("-------------------------------------------------------") 
print("PREVIEW MODE:") 
print("  Press [W] / [S] : Increase / Decrease Exposure (Brightness)") 
print("  Press [A]       : Make WHITER / COOLER (Lowers Kelvin)") 
print("  Press [D]       : Make WARMER / YELLOW (Increases Kelvin)") 
print("  Press [E] / [C] : Boost / Lower Sensor Gain") 
print("  Press [SPACE]   : LOCK settings and START 30s RECORDING") 
print("  Press [Q]       : QUIT") 
print("-------------------------------------------------------") 

# 1. Calibration / Preview Loop
while True:
    ret, frame = cap.read() 
    if not ret or frame is None: 
        continue 

    if len(frame.shape) == 3 and frame.shape[2] == 2: 
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV) 
    else: 
        frame_bgr = frame 

    display = frame_bgr.copy() 
    cv2.putText(display, f"Exp: {exposure_val:.1f} (W/S) | WB: {wb_temp}K (A/D) | Gain: {gain_val} (E/C)", 
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2) 
    cv2.putText(display, "Press 'A' to shift cooler/whiter. SPACE to record.", 
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2) 
    cv2.imshow("Camera Calibration", display) 

    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'): 
        cap.release() 
        cv2.destroyAllWindows() 
        exit() 
    elif key == ord('w'): 
        exposure_val = min(-1.0, exposure_val + 0.5) 
        apply_camera_settings(exposure_val, wb_temp, brightness_val, gain_val) 
    elif key == ord('s'): 
        exposure_val = max(-10.0, exposure_val - 0.5) 
        apply_camera_settings(exposure_val, wb_temp, brightness_val, gain_val) 
    elif key == ord('a'): 
        wb_temp = max(2000, wb_temp - 200) 
        apply_camera_settings(exposure_val, wb_temp, brightness_val, gain_val) 
    elif key == ord('d'): 
        wb_temp = min(10000, wb_temp + 200) 
        apply_camera_settings(exposure_val, wb_temp, brightness_val, gain_val) 
    elif key == ord('e'): 
        gain_val = min(100, gain_val + 5) 
        apply_camera_settings(exposure_val, wb_temp, brightness_val, gain_val) 
    elif key == ord('c'): 
        gain_val = max(0, gain_val - 5) 
        apply_camera_settings(exposure_val, wb_temp, brightness_val, gain_val) 
    elif key == 32:  
        break 

cv2.destroyWindow("Camera Calibration") 

# --- NEW FOLDER AUTOMATION ---
base_dir = r"C:\Users\tolis\Desktop\Praktiki\rPPG_project\experiment" 
os.makedirs(base_dir, exist_ok=True)

# Find the highest existing PXXX folder
existing_folders = [f for f in os.listdir(base_dir) if f.startswith("P") and f[1:].isdigit()]
if existing_folders:
    latest_num = max([int(f[1:]) for f in existing_folders])
    next_num = latest_num + 1
else:
    next_num = 1

session_id = f"P{next_num:03d}"
session_dir = os.path.join(base_dir, session_id)
os.makedirs(session_dir, exist_ok=True)

print(f"\nSettings locked! Recording 30 seconds into RAM for session {session_id}...")

# 2. Recording Loop (Straight into RAM)
frames_buffer = [] 

# Capture the exact start time of the recording
recording_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
start_time = time.time() 

while (time.time() - start_time) < 30.0: 
    ret, frame = cap.read() 
    if not ret or frame is None: 
        break 
    frames_buffer.append(frame) 

# 3. Calculate TRUE Frames Per Second
actual_duration = time.time() - start_time 
frames_captured = len(frames_buffer) 
true_fps = frames_captured / actual_duration 

print(f"Captured {frames_captured} frames in {actual_duration:.2f}s (True FPS: {true_fps:.2f}).") 
print("Writing to disk... Do not close.") 

# 4. Write Frames to File
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

# Update output path to route inside the new session directory
output_video_path = os.path.join(session_dir, f"{session_id}_raw.avi")
fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
out = cv2.VideoWriter(output_video_path, fourcc, true_fps, (width, height)) 

for f in frames_buffer: 
    if len(f.shape) == 3 and f.shape[2] == 2: 
        f = cv2.cvtColor(f, cv2.COLOR_YUV2BGR_YUYV) 
    out.write(f) 

out.release() 

# Save companion JSON with verified hardware rate and new start time
metadata = {
    "session_id": session_id,
    "start_time": recording_start_time,
    "true_fps": float(true_fps), 
    "frames_captured": int(frames_captured), 
    "duration_sec": float(actual_duration) 
}

json_path = os.path.join(session_dir, f"{session_id}_metadata.json")
with open(json_path, "w") as f: 
    json.dump(metadata, f, indent=4) 
print(f"Saved recording metadata to: {json_path}") 

# 5. Cleanup
cap.release() 
cv2.destroyAllWindows() 
print("Done saving!")
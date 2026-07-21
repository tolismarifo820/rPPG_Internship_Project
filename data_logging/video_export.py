# To call in main:
# from data_logging.video_export import *

import os
import cv2
from data_logging.csv_logger import create_participant_folder

def open_acquisition_video_writer(participant_id, frame_width, frame_height, fps_value):
    try:
        folder = create_participant_folder(participant_id)
        video_path = os.path.join(folder, "acquisition_30s.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, float(fps_value), (int(frame_width), int(frame_height)))
        if not writer.isOpened():
            video_path = os.path.join(folder, "acquisition_30s.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(video_path, fourcc, float(fps_value), (int(frame_width), int(frame_height)))
        if not writer.isOpened(): return None, ""
        return writer, video_path
    except Exception: return None, ""

def close_acquisition_video_writer(video_writer):
    try:
        if video_writer is not None: video_writer.release()
    except Exception: pass
# To call in main:
# from gui.main_window import *

import cv2
import numpy as np
import time
import os
from config import *

def display_metric_value(face_detected_now: bool, vitals_enabled: bool, value):
    if not face_detected_now: return None
    if not vitals_enabled: return "Calculating..."
    return value

def apply_fade(value, alpha_value: float):
    if value is None: return None
    if isinstance(value, str): return value
    return float(value) * alpha_value

def load_logo_image(path: str, target_w: int, target_h: int):
    try:
        if not path or not os.path.exists(path): return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        h, w = img.shape[:2]
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception: return None

def overlay_image_center(canvas, img, box_x, box_y, box_w, box_h):
    if img is None: return
    ih, iw = img.shape[:2]
    x, y = max(box_x, box_x + (box_w - iw) // 2), max(box_y, box_y + (box_h - ih) // 2)
    roi = canvas[y:y + ih, x:x + iw]
    if roi.shape[0] != ih or roi.shape[1] != iw: return
    if img.shape[2] == 4:
        alpha_img = img[:, :, 3].astype(np.float32) / 255.0
        alpha_img = alpha_img[:, :, None]
        fg, bg = img[:, :, :3].astype(np.float32), roi.astype(np.float32)
        out = fg * alpha_img + bg * (1.0 - alpha_img)
        canvas[y:y + ih, x:x + iw] = np.clip(out, 0, 255).astype(np.uint8)
    else: canvas[y:y + ih, x:x + iw] = img

def wrap_text_to_width(text, font, font_scale, thickness, max_width):
    words = text.split()
    if not words: return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        test = current + " " + word
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_width: current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines

# ========================================
# UI Drawing logic
# ========================================
def draw_card(canvas, target_box, title, value, unit, color, show_pulse_icon=False, current_bpm=None):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)
    cv2.putText(canvas, title, (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, TITLE_COLOR, 2, cv2.LINE_AA)

    if isinstance(value, str): val_str, font_scale, thickness = value, 1.0, 2
    elif value is not None and value > 0: val_str, font_scale, thickness = f"{value:.0f}", 2.4, 3
    else: val_str, font_scale, thickness = "--", 2.4, 3

    (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    text_x, text_y = x + max(24, (w - tw) // 2), y + h // 2 + th // 2 + 10
    cv2.putText(canvas, val_str, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)

    if unit: cv2.putText(canvas, unit, (x + 24, y + h - 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, SUBTLE_TEXT, 1, cv2.LINE_AA)
    if show_pulse_icon and isinstance(current_bpm, (int, float, np.floating)) and current_bpm > 30:
        bps = float(current_bpm) / 60.0
        pulse_scale = 0.75 + 0.25 * np.sin((time.time() * bps * 2 * np.pi) % (2 * np.pi))
        heart_x, heart_y = x + w - 52, y + 56
        pts = (np.array([[0, -8], [8, -15], [15, -8], [0, 12], [-15, -8], [-8, -15]], np.int32) * pulse_scale * 1.4).astype(np.int32)
        cv2.fillPoly(canvas, [pts + [heart_x, heart_y]], color, lineType=cv2.LINE_AA)

def draw_waveform_card(canvas, target_box, wave_buffer, method_name):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)
    
    # Use a slightly smaller font scale or short format if the title is long
    title = f"PPG ({method_name})" 
    cv2.putText(canvas, title, (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, TITLE_COLOR, 2, cv2.LINE_AA)

    if len(wave_buffer) > 10:
        plot_data = np.array(list(wave_buffer), dtype=np.float32)
        plot_data = plot_data - np.mean(plot_data)
        d_min, d_max = plot_data.min(), plot_data.max()
        norm = (plot_data - d_min) / (d_max - d_min) if d_max > d_min + 1e-5 else np.full_like(plot_data, 0.5)
        norm = np.clip(norm, 0.0, 1.0)
            
        plot_h, plot_w = h - 85, w - 50
        for i in range(len(norm) - 1):
            p1_x, p1_y = x + 24 + int(i * plot_w / 100), y + 65 + int((1 - norm[i]) * plot_h)
            p2_x, p2_y = x + 24 + int((i + 1) * plot_w / 100), y + 65 + int((1 - norm[i + 1]) * plot_h)
            cv2.line(canvas, (p1_x, p1_y), (p2_x, p2_y), ACCENT_PULSE, 2, cv2.LINE_AA)

def draw_protocol_timer_card(canvas, target_box, protocol_state, remaining_seconds):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)

    if protocol_state == PROTOCOL_LOCKING:
        cv2.putText(canvas, "STABILIZING", (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.0, TITLE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "WAIT", (x + 86, y + h // 2 + 12), cv2.FONT_HERSHEY_DUPLEX, 1.9, ACCENT_WARNING, 3, cv2.LINE_AA)
        cv2.putText(canvas, "Keep still until vitals activate", (x + 22, y + h - 34), cv2.FONT_HERSHEY_DUPLEX, 0.52, SUBTLE_TEXT, 1, cv2.LINE_AA)
    elif protocol_state == PROTOCOL_ACQUISITION:
        cv2.putText(canvas, "ACQUISITION TIMER", (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.95, TITLE_COLOR, 2, cv2.LINE_AA)
        timer_text = f"00:{max(0, int(round(remaining_seconds))):02d}"
        (tw, th), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_DUPLEX, 2.25, 4)
        cv2.putText(canvas, timer_text, (x + (w - tw) // 2, y + h // 2 + th // 2), cv2.FONT_HERSHEY_DUPLEX, 2.25, ACCENT_INFO, 4, cv2.LINE_AA)
        cv2.putText(canvas, "Please observe the GUI", (x + 36, y + h - 34), cv2.FONT_HERSHEY_DUPLEX, 0.62, SUBTLE_TEXT, 1, cv2.LINE_AA)
    elif protocol_state == PROTOCOL_COMPLETE:
        cv2.putText(canvas, "ACQUISITION", (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.05, TITLE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "COMPLETE", (x + 48, y + h // 2 + 12), cv2.FONT_HERSHEY_DUPLEX, 1.7, ACCENT_OK, 3, cv2.LINE_AA)
        cv2.putText(canvas, "Press N for next participant", (x + 24, y + h - 34), cv2.FONT_HERSHEY_DUPLEX, 0.55, SUBTLE_TEXT, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "POSITIONING", (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.05, TITLE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "READY", (x + 72, y + h // 2 + 12), cv2.FONT_HERSHEY_DUPLEX, 1.8, ACCENT_WARNING, 3, cv2.LINE_AA)
        cv2.putText(canvas, "Press SPACE to start", (x + 42, y + h - 34), cv2.FONT_HERSHEY_DUPLEX, 0.62, SUBTLE_TEXT, 1, cv2.LINE_AA)

def draw_logo_card(canvas, target_box, logo_img, freqs, fftAvg, minFrequency_val, maxFrequency_val, current_hr):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)
    
    # Shortened title to fit comfortably within the top-right box bounds
    cv2.putText(canvas, "SPECTRUM", (x + 24, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.0, TITLE_COLOR, 2, cv2.LINE_AA)

    valid = (freqs >= minFrequency_val) & (freqs <= maxFrequency_val)
    if not np.any(valid): return

    f, mag = freqs[valid], fftAvg[valid]
    if len(mag) < 2: return

    plot_x, plot_y, plot_w, plot_h = x + 24, y + 75, w - 48, h - 95
    
    # Safe robust normalization to prevent flatlining/clipping
    mag_min, mag_max = np.min(mag), np.max(mag)
    if mag_max > mag_min + 1e-4:
        norm_mag = (mag - mag_min) / (mag_max - mag_min)
    else:
        norm_mag = np.zeros_like(mag)
    norm_mag = np.clip(norm_mag, 0.0, 1.0)

    points = [(plot_x + int((i / (len(f) - 1)) * plot_w), plot_y + plot_h - int(norm_mag[i] * plot_h)) for i in range(len(f))]
    for i in range(len(points) - 1): 
        cv2.line(canvas, points[i], points[i+1], ACCENT_BREATH, 2, cv2.LINE_AA)

    if current_hr is not None and current_hr > 0:
        hr_hz = current_hr / 60.0
        if minFrequency_val <= hr_hz <= maxFrequency_val:
            hx = plot_x + int(((hr_hz - minFrequency_val) / (maxFrequency_val - minFrequency_val)) * plot_w)
            cv2.line(canvas, (hx, plot_y), (hx, plot_y + plot_h), ACCENT_PULSE, 1, cv2.LINE_AA)
            cv2.putText(canvas, "PEAK", (hx - 15, plot_y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.45, ACCENT_PULSE, 1, cv2.LINE_AA)

def draw_camera_card(canvas, target_box, frame, evm_roi_bbox=None, mask_contours=None):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cam_h, cam_w = h - 20, w - 20
    canvas[y + 10:y + 10 + cam_h, x + 10:x + 10 + cam_w] = cv2.resize(frame, (cam_w, cam_h))
    scale_x, scale_y = cam_w / realWidth, cam_h / realHeight

    if evm_roi_bbox is not None:
        ex1, ey1, ex2, ey2 = evm_roi_bbox
        cv2.rectangle(canvas, (int(ex1 * scale_x) + x + 10, int(ey1 * scale_y) + y + 10), (int(ex2 * scale_x) + x + 10, int(ey2 * scale_y) + y + 10), ACCENT_EVM, 2, cv2.LINE_AA)
    
    if mask_contours is not None:
        scaled_contours = []
        for cnt in mask_contours:
            scaled_cnt = np.zeros_like(cnt)
            scaled_cnt[:, 0, 0] = cnt[:, 0, 0] * scale_x + x + 10
            scaled_cnt[:, 0, 1] = cnt[:, 0, 1] * scale_y + y + 10
            scaled_contours.append(scaled_cnt)
        cv2.polylines(canvas, scaled_contours, isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)
    cv2.putText(canvas, "CV.WINDOW", (x + 24, y + 46), cv2.FONT_HERSHEY_DUPLEX, 1.0, TITLE_COLOR, 2, cv2.LINE_AA)

def draw_participant_id(canvas, participant_id):
    # Moved down by changing y offset from + PAD to + 55
    x = BOX_STATUS[0] + BOX_STATUS[2] - 310 - PAD
    y = BOX_STATUS[1] + 55  
    w, h = 310, 34
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), ACCENT_INFO, 2)
    cv2.putText(canvas, f"PARTICIPANT: {participant_id}", (x + 16, y + 23), cv2.FONT_HERSHEY_DUPLEX, 0.58, TITLE_COLOR, 1, cv2.LINE_AA)

def draw_acquisition_control(canvas, active: bool):
    label = "ACQUISITION: ON" if active else "ACQUISITION: PAUSED"
    hint = "SPACE: Toggle | M: Method"
    color = ACCENT_OK if active else ACCENT_WARNING
    
    # Slightly widen the box container to fit everything cleanly
    x, y, w, h = CANVAS_W - 350, CANVAS_H - 42, 330, 30
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    cv2.circle(canvas, (x + 15, y + 15), 6, color, -1, cv2.LINE_AA)
    
    cv2.putText(canvas, label, (x + 28, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, TITLE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, hint, (x + 165, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.38, SUBTLE_TEXT, 1, cv2.LINE_AA)

def draw_status_icon(canvas, kind, cx, cy):
    color = {"ok": ACCENT_OK, "info": ACCENT_INFO, "warn": ACCENT_WARNING, "error": ACCENT_ERROR}.get(kind, TITLE_COLOR)
    cv2.circle(canvas, (cx, cy), 10, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 1, cv2.LINE_AA)
    return color

def build_status_items(face_detected_now, vitals_enabled, stable_duration, acquisition_active, signal_quality_label, signal_quality_score, protocol_state, remaining_seconds, last_reset_reason, last_completed_valid, last_completed_mean_sqi):
    items = []
    if protocol_state == PROTOCOL_COMPLETE:
        if last_completed_valid is False: items.append(("warn", "Low-quality acquisition", f"Mean SQI: {last_completed_mean_sqi:.0f}/100. Consider repeating the measurement."))
        elif last_completed_valid is True: items.append(("ok", "Valid acquisition complete", f"30-second recording finished. Mean SQI: {last_completed_mean_sqi:.0f}/100."))
        else: items.append(("ok", "Acquisition complete", "The 30-second recording has finished and CSV files have been closed."))
        items.append(("info", "Next step", "Press N to create the next participant, or SPACE to repeat acquisition for this participant."))
        return items

    if not acquisition_active:
        items.append(("info", "Position participant", "Face detection is active. Place the participant correctly inside the camera window."))
        if last_reset_reason: items.append(("error", "Acquisition reset", last_reset_reason))
        items.append(("warn", "Acquisition paused", "Press SPACE to begin signal stabilization."))
    elif protocol_state == PROTOCOL_LOCKING:
        items.append(("warn", "Stabilizing signal", "Please remain still. The 30-second recording will start automatically when vitals become active."))
    else: items.append(("info", "Acquisition in progress", f"Participant observes the GUI. Remaining time: {max(0, remaining_seconds):.0f}s."))

    if signal_quality_label is not None and signal_quality_score is not None:
        color_code = "ok" if "GOOD" in signal_quality_label else "warn" if "FAIR" in signal_quality_label else "error"
        items.append((color_code, f"{signal_quality_label}: {signal_quality_score:.0f}/100", "Quality estimate based on movement, lighting stability, and pulse-like periodicity."))

    if not face_detected_now: items.append(("error", "No Face Detected", "Place your face inside the CV window / EVM ROI."))
    elif not vitals_enabled:
        remaining = max(0.0, FACE_STABLE_SECONDS_REQUIRED - stable_duration)
        items.append(("info", "Face detected inside ROI", "Detection is valid and the face is inside the EVM ROI."))
        items.append(("warn", f"Face locking: {remaining:.1f}s remaining", "Keep still, face the camera, and avoid sudden head motion."))
    else: items.append(("ok", "Vitals active", "Stable segment acquired. Live values are being displayed and logged."))
    return items

def draw_status_card(canvas, target_box, status_items):
    x, y, w, h = target_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_COLOR, 2)
    cv2.putText(canvas, "STATUS / MESSAGES", (x + 30, y + 58), cv2.FONT_HERSHEY_DUPLEX, 1.25, TITLE_COLOR, 2, cv2.LINE_AA)

    content_x, content_y = x + 34, y + 128
    card_gap, card_h, inner_w = 14, 70, w - 68

    for kind, title, body in status_items[:4]:
        if content_y + card_h > y + h - 16: break
        panel_color = (82, 82, 82) if ACTIVE_THEME_NAME == "dark" else (252, 252, 252)
        if kind == "error": panel_color = (74, 74, 86) if ACTIVE_THEME_NAME == "dark" else (245, 240, 245)
        
        cv2.rectangle(canvas, (content_x, content_y), (content_x + inner_w, content_y + card_h), panel_color, -1)
        cv2.rectangle(canvas, (content_x, content_y), (content_x + inner_w, content_y + card_h), BORDER_COLOR, 1)
        color = draw_status_icon(canvas, kind, content_x + 18, content_y + 20)
        cv2.putText(canvas, title, (content_x + 38, content_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.72, color, 1, cv2.LINE_AA)

        wrapped = wrap_text_to_width(body, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1, inner_w - 52)
        line_y = content_y + 50
        for line in wrapped[:2]:
            cv2.putText(canvas, line, (content_x + 18, line_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, TITLE_COLOR, 1, cv2.LINE_AA)
            line_y += 18
        content_y += card_h + card_gap
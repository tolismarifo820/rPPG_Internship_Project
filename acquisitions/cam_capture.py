import cv2
import time

def get_camera_metadata(cap):
    """Returns a dictionary of current camera properties."""
    return {
        "Width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "Height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "FPS": cap.get(cv2.CAP_PROP_FPS),
        "Brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "Contrast": cap.get(cv2.CAP_PROP_CONTRAST),
        "Exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "Auto_Exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
    }

def decode_fourcc(v):
    """Convert OpenCV FOURCC integer to string."""
    try:
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return "UNKN"

def encode_fourcc(c1, c2, c3, c4):
    """Convert string to OpenCV FOURCC integer."""
    return cv2.VideoWriter_fourcc(c1, c2, c3, c4)

def measure_true_fps(cap, num_frames=15):
    """Measure actual FPS by timing frame grabs."""
    start_time = time.perf_counter()
    for _ in range(num_frames):
        ret, _ = cap.read()
        if not ret:
            return 0.0
    elapsed = time.perf_counter() - start_time
    return num_frames / elapsed if elapsed > 0 else 0.0

def test_all_cameras():
    print("=" * 65)
    print("SCANNING FOR ALL CONNECTED USB CAMERAS")
    print("=" * 65)

    # 1. Discover all connected cameras
    available_cameras = []
    for idx in range(6):  # Checks indices 0 to 5
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(idx)
            cap.release()

    if not available_cameras:
        print("[-] No active cameras detected on any USB port.")
        return

    print(f"[+] Found {len(available_cameras)} active camera(s): Indices {available_cameras}\n")

    test_resolutions = [(640, 360), (640, 480), (800, 600), (1280, 720), (1920, 1080)]
    test_fourccs = ['YUY2', 'MJPG', 'YUYV', 'NV12']

    # 2. Run the diagnostic inspect on EACH camera found
    for camera_index in available_cameras:
        print("\n" + "=" * 65)
        print(f"INSPECTING CAMERA AT INDEX: {camera_index}")
        print("=" * 65)

        results = []

        print("\n[*] Testing Resolutions and Pixel Formats (This takes a few seconds)...")
        for width, height in test_resolutions:
            for fmt in test_fourccs:
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    continue
                    
                fourcc_int = encode_fourcc(*fmt)
                cap.set(cv2.CAP_PROP_FOURCC, fourcc_int)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                time.sleep(0.1)
                
                actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
                claimed_fps = cap.get(cv2.CAP_PROP_FPS)
                
                is_match = (actual_w == width and actual_h == height and actual_fourcc == fmt)
                
                if is_match:
                    true_fps = measure_true_fps(cap, num_frames=15)
                    results.append({
                        "Requested": f"{width}x{height} {fmt}",
                        "Actual Res": f"{int(actual_w)}x{int(actual_h)}",
                        "Format": actual_fourcc,
                        "Claimed FPS": round(claimed_fps, 2),
                        "True FPS": round(true_fps, 2)
                    })
                
                cap.release()

        if results:
            print("\n--- Supported Configurations ---")
            header = f"{'Requested':<20} | {'Actual Res':<12} | {'Format':<8} | {'Claimed FPS':<12} | {'True FPS':<10}"
            print(header)
            print("-" * len(header))
            for r in results:
                print(f"{r['Requested']:<20} | {r['Actual Res']:<12} | {r['Format']:<8} | {r['Claimed FPS']:<12} | {r['True FPS']:<10}")
        else:
            print("\n[-] Could not determine any valid configurations.")

        print("\n[*] Checking Current Camera Properties...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index} for properties check.")
            continue

        properties = [
            ("Brightness", cv2.CAP_PROP_BRIGHTNESS),
            ("Contrast", cv2.CAP_PROP_CONTRAST),
            ("Saturation", cv2.CAP_PROP_SATURATION),
            ("Hue", cv2.CAP_PROP_HUE),
            ("Gain", cv2.CAP_PROP_GAIN),
            ("Exposure", cv2.CAP_PROP_EXPOSURE),
            ("Auto Exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
            ("Gamma", cv2.CAP_PROP_GAMMA),
            ("Sharpness", cv2.CAP_PROP_SHARPNESS),
            ("Temperature", cv2.CAP_PROP_TEMPERATURE),
            ("Focus", cv2.CAP_PROP_FOCUS),
            ("Auto Focus", cv2.CAP_PROP_AUTOFOCUS),
            ("Zoom", cv2.CAP_PROP_ZOOM),
        ]

        print("-" * 40)
        for name, prop in properties:
            try:
                value = cap.get(prop)
                print(f"{name:20}: {value}")
            except:
                print(f"{name:20}: Not Supported")
        print("-" * 40)

        print(f"\n[*] Starting Live Preview for Camera {camera_index}...")
        print("    -> Press 'Q' to quit and move to the next camera (if any).")
        print("    -> NO DATA IS BEING RECORDED OR SAVED.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed.")
                break

            cv2.putText(
                frame,
                f"Cam {camera_index}: {frame.shape[1]}x{frame.shape[0]} | NO RECORDING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255), 
                2,
            )

            cv2.imshow(f"Camera {camera_index} Inspector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_all_cameras()
# rPPG & EVM Dashboard

This repository contains the source code for a robust, real-time remote photoplethysmography (rPPG) and Eulerian Video Magnification (EVM) dashboard. It was developed as part of an internship project at the **CENEBIT** (Center of Excellence in Biomedical/Bioengineering and IT) at **AUTH** (Aristotle University of Thessaloniki).

Designed primarily for deployment on a Computer but it includes a modular architecture also for a Raspberry Pi (utilizing the PiCamera2 interface or standard webcams), the system extracts vital signs (Heart Rate, Respiration Rate, SpO2) using purely optical methods.

---

## ✨ Key Features

* **Robust Face Tracking & ROI Extraction:** Utilizes Google's MediaPipe FaceMesh to explicitly extract highly stable Regions of Interest (ROI), isolating the forehead and cheeks for accurate signal extraction while ignoring background noise.
* **Multi-Method rPPG & Advanced Extraction:** Press a single key to cycle between state-of-the-art rPPG algorithms in real-time:
* **GREEN:** Single-channel green extraction.
* **CHROM:** Chrominance-based extraction.
* **POS:** Plane-Orthogonal-to-Skin extraction.
* **ICA / PCA:** Blind source separation methods.
* **PBV (Projected Blood Volume):** Leverages a blood-volume signature vector to project chrominance signals, maximizing signal-to-noise ratio against motion artifacts.
* **SSR (Spatial-Subspace Rotation):** Utilizes spatial redundancies across facial regions to stabilize pulse extraction against subtle head movements.
* **LGI (Local Group Invariance):** Enhances robustness by exploiting local spatial intensity relationships across skin patches, minimizing illumination variations.


* **Eulerian Video Magnification (EVM):** Features a dynamic Gaussian pyramid implementation with BGR and YIQ mode toggles to visually amplify micro-color changes in the skin caused by blood flow.
* **Bento Grid UI:** A fully custom, highly responsive OpenCV-based dashboard featuring waveform plotting, FFT spectrums, Signal Quality Index (SQI) monitoring, and light/dark theme toggles.
* **Cloud Ready:** Includes a `main.ipynb` Jupyter Notebook configured to run headless pipeline processing on Google Colab using uploaded video files.

---

## 📂 Project Structure

```text
rPPG_Internship_Project/
├── acquisitions/                # Camera hardware initialization and capture handling
├── data_logging/                # CSV metadata, video exporting, and session visualization
├── estimation/                  # FFT-based BPM estimation and SQI logic
├── face/                        # Facial tracking utility scripts
├── gui/                         # Bento Grid UI components and waveform plotters
├── methods/                     # rPPG extraction algorithms (CHROM, POS, GREEN, PBV, etc.)
├── plots/                       # Output directory for waveform visualizations
├── roi/                         # MediaPipe FaceMesh processing and masks
├── rPPG_site/                   # Web application prototypes
├── signals/                     # DSP filtering and preprocessing (Butterworth, etc.)
├── tests/                       # Unit tests and isolated script testing
├── logs/                        # Local session exports (CSVs, metadata, ML images)
├── logs_tif90/                  # Directory for Raspberry Pi TIF files
├── main.py                      # The main application and UI loop
├── main.ipynb                   # Google Colab-compatible notebook
├── config.py                    # Global configurations and UI theme data
├── requirements.txt             # Project dependencies for pip installation
├── after_evm.py                 # Post-magnification processing utilities
├── evaluate_methods.py          # Script for performance evaluation of extraction methods
├── extraction_from_same_video.py# Batch processing script for video files
├── extraction2.py               # Alternative pipeline extraction script
├── record_raw_video.py          # Utility script for recording raw video inputs
├── *.m                          # MATLAB analysis scripts (Bicoherence.m, CWT.m, Wavelet.m, etc.)
└── tif90*.py                    # Hardware-specific optimization scripts for Raspberry Pi

```

---

## 🛠️ Prerequisites

* Python 3.9+
* A connected webcam or Raspberry Pi Camera Module.

---

## 🚀 Setup Instructions

**1. Clone the repository and navigate to the project directory:**

```bash
git clone <repository_url>
cd rPPG_project

```

**2. Create a virtual environment (Recommended):**
Isolating dependencies ensures this project does not interfere with other Python projects on your machine.

```bash
python -m venv venv

```

**3. Activate the virtual environment:**

* **Windows:**
```bash
.\venv\Scripts\activate

```


* **macOS/Linux:**
```bash
source venv/bin/activate

```



**4. Install dependencies:**
Install the required libraries matching the exact environment configurations:

```bash
pip install -r requirements.txt

```

---

## 🏃 Running the Application

Execute the main script from the root of the project directory:

```bash
python main.py

```

### Keyboard Controls

The graphical interface is controlled entirely via the following keyboard shortcuts while the window is in focus:

| Key | Action | Description |
| --- | --- | --- |
| `SPACE` | **Start/Stop Acquisition** | Toggles the state machine between Positioning, Locking, and Active Acquisition. Saves data upon valid completion. |
| `m` | **Cycle rPPG Method** | Switches between extraction algorithms (e.g., POS, CHROM, GREEN, ICA, PCA, LGI, PBV) and fused combinations. |
| `r` | **Cycle ROI** | Changes the facial region used for extraction (Full Face, Forehead, Cheeks, Segmented). |
| `e` | **Cycle EVM Mode** | Toggles Eulerian Video Magnification processing spaces (e.g., BGR, YIQ). |
| `n` | **New Participant** | Resets the session, clears buffers, and creates a new localized logging directory for a new subject. |
| `t` | **Toggle Theme** | Switches the UI between Light and Dark modes. |
| `q` | **Quit** | Safely releases the camera, closes files, and exits the application. |

---

## 📊 Outputs & Data Logging

When an acquisition is successfully completed (or reset), the system generates a unique participant folder within the `logs/` directory containing:

* **`metadata.csv`**: Frame-by-frame calculations, timestamps, HR/RR/SpO2 values, SQI (Signal Quality Index) metrics, and bounding box coordinates.
* **`hardware_config.json`**: The locked camera hardware settings (exposure, white balance) used during the session.
* Visual plots and optionally exported video/frame sequences for downstream Machine Learning tasks.

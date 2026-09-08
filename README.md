#  Raspberry Pi rPPG & EVM Workshop Dashboard

This repository contains the source code for a robust, real-time remote photoplethysmography (rPPG) and Eulerian Video Magnification (EVM) dashboard. It was developed as part of an internship project at the **CENEBIT** (Center of Excellence in Biomedical/Bioengineering and IT) at **AUTH** (Aristotle University of Thessaloniki).

Designed primarily for deployment on a Raspberry Pi (utilizing the PiCamera2 interface or standard webcams), the system extracts vital signs (Heart Rate, Respiration Rate, SpO2) using purely optical methods.

---

## ✨ Key Features

* **Robust Face Tracking & ROI Extraction:** Utilizes Google's MediaPipe FaceMesh to explicitly extract highly stable Regions of Interest (ROI), isolating the forehead and cheeks for accurate signal extraction while ignoring background noise.
* **Multi-Method rPPG:** Press a single key to cycle between state-of-the-art rPPG algorithms in real-time:
  * **GREEN:** Single-channel green extraction.
  * **CHROM:** Chrominance-based extraction.
  * **POS:** Plane-Orthogonal-to-Skin extraction.
  * **ICA / PCA:** Blind source separation methods.
* **Eulerian Video Magnification (EVM):** Features a dynamic Gaussian pyramid implementation with BGR and YIQ mode toggles to visually amplify micro-color changes in the skin caused by blood flow.
* **Bento Grid UI:** A fully custom, highly responsive OpenCV-based dashboard featuring waveform plotting, FFT spectrums, Signal Quality Index (SQI) monitoring, and light/dark theme toggles.
* **Cloud Ready:** Includes a `main.ipynb` Jupyter Notebook configured to run headless pipeline processing on Google Colab using uploaded video files.

---

## 📂 Project Structure

```text
rPPG_Internship_Project/
├── main.py                # The main application and UI loop
├── main.ipynb             # Google Colab-compatible notebook
├── config.py              # Global configurations and UI theme data
├── README.md              # Project documentation
├── methods/               # rPPG extraction algorithms (CHROM, POS, etc.)
├── roi/                   # MediaPipe FaceMesh processing and masks
├── signals/               # DSP filtering and preprocessing (Butterworth, etc.)
├── estimation/            # FFT-based BPM estimation and SQI logic
├── gui/                   # Bento Grid UI components and waveform plotters
├── logs/                  # Ignored folder for generated CSV metadata and outputs
└── logs_tif90/            # Directory for Raspberry Pi TIF files

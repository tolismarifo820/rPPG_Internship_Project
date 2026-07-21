# rPPG Internship Project

## Overview
This repository contains a modular Python pipeline for remote photoplethysmography (rPPG). The software captures face video feeds, isolates regions of interest (ROIs), extracts raw RGB signals, and applies various signal processing algorithms to estimate heart rate (HR) continuously. 

This project was developed during a summer research internship at the Medical Physics and Digital Innovation Lab (Aristotle University of Thessaloniki) and the Centre for Neurosciences & Biomedical Technology, under the supervision of Dr. Alkinous Athanasiou.

## Features
* **Live & Recorded Video Processing**: Modular acquisition components to handle direct camera feeds or pre-recorded video analysis.
* **Face Tracking & ROI Extraction**: Includes algorithms for detecting facial landmarks, skin segmentation, and extracting specific regions like cheeks, forehead, or the full face.
* **Multiple rPPG Algorithms**: 
  * GREEN
  * CHROM
  * POS
  * ICA-based (using `scikit-learn` FastICA)
  * PCA-based (using `scikit-learn` PCA)
* **Real-time GUI**: A built-in user interface for controlling the pipeline and plotting HR signals dynamically.
* **Data Logging**: Built-in CSV and metadata logging for post-analysis and clinical validation (e.g., participant registries, workshop statistics).

## Project Structure
The codebase has been refactored from a monolithic script into a highly modular architecture:

```text
rPPG_Internship_Project/
│
├── main.py                     # Main execution script 
├── config.py                   # Configuration and hyperparameter management
│
├── acquisitions/               # Camera capture and video handling modules
├── data_logging/               # CSV, video export, and metadata loggers
├── estimation/                 # Heart rate estimation (FFT, Autocorrelation, Peak detection, SQI)
├── face/                       # Face detection, landmarks, and skin segmentation
├── gui/                        # Main window, dynamic plotting, and UI controls
├── methods/                    # Core rPPG algorithms (CHROM, GREEN, POS, ICA, PCA)
├── roi/                        # ROI manager and spatial extraction strategies
├── signals/                    # RGB extraction, buffering, preprocessing, and filtering
└── tests/                      # Unit testing suite

Author
Apostolos Marifoglou
Student, School of Electrical and Computer Engineering
Aristotle University of Thessaloniki (AUTH)

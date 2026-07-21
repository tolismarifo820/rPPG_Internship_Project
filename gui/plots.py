# To call in main:
# from gui.plots import *

import os
import csv
import numpy as np
from config import MATPLOTLIB_AVAILABLE

if MATPLOTLIB_AVAILABLE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

def export_session_plot(participant_folder, raw_csv_path):
    if not MATPLOTLIB_AVAILABLE or not os.path.exists(raw_csv_path): return
    try:
        timestamps = []
        hr_vals, rr_vals, spo2_vals, rppg_vals = [], [], [], []
        r_vals, g_vals, b_vals = [], [], []
        
        with open(raw_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only collect data during the actual 30s acquisition window
                if row.get("ProtocolPhase") == "ACQUISITION":
                    try:
                        t = float(row.get("PhaseElapsed", 0))
                        
                        # Parse strings safely, defaulting to NaN if the row is empty/invalid
                        hr = float(row.get("HR_Display")) if row.get("HR_Display") not in ("", None, "None") else np.nan
                        rr = float(row.get("RR_Raw")) if row.get("RR_Raw") not in ("", None, "None") else np.nan
                        spo2 = float(row.get("SpO2_Raw")) if row.get("SpO2_Raw") not in ("", None, "None") else np.nan
                        rppg = float(row.get("rPPG_Chrom")) if row.get("rPPG_Chrom") not in ("", None, "None") else np.nan
                        
                        r = float(row.get("Mean_R")) if row.get("Mean_R") not in ("", None, "None") else np.nan
                        g = float(row.get("Mean_G")) if row.get("Mean_G") not in ("", None, "None") else np.nan
                        b = float(row.get("Mean_B")) if row.get("Mean_B") not in ("", None, "None") else np.nan

                        timestamps.append(t)
                        hr_vals.append(hr)
                        rr_vals.append(rr)
                        spo2_vals.append(spo2)
                        rppg_vals.append(rppg)
                        r_vals.append(r)
                        g_vals.append(g)
                        b_vals.append(b)
                    except ValueError:
                        continue
        
        if len(timestamps) < 5: return
        
        # Create a multi-plot figure containing 5 stacked charts
        fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Participant Session Summary", fontsize=16, fontweight="bold")
        
        # 1. RGB Signals
        axs[0].plot(timestamps, r_vals, color='#ff3333', label="Mean Red", alpha=0.8)
        axs[0].plot(timestamps, g_vals, color='#33cc33', label="Mean Green", alpha=0.8)
        axs[0].plot(timestamps, b_vals, color='#3366ff', label="Mean Blue", alpha=0.8)
        axs[0].set_ylabel("RGB Intensity")
        axs[0].legend(loc="upper right")
        axs[0].grid(True, linestyle="--", alpha=0.5)
        
        # 2. rPPG Signal
        axs[1].plot(timestamps, rppg_vals, color='purple', label="Filtered rPPG Waveform")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend(loc="upper right")
        axs[1].grid(True, linestyle="--", alpha=0.5)
        
        # 3. Heart Rate
        axs[2].plot(timestamps, hr_vals, color='#ff5050', label="Heart Rate (BPM)", linewidth=2.5)
        axs[2].set_ylabel("BPM")
        axs[2].legend(loc="upper right")
        axs[2].grid(True, linestyle="--", alpha=0.5)
        
        # 4. Respiratory Rate
        axs[3].plot(timestamps, rr_vals, color='#3399ff', label="Respiratory Rate (BR/MIN)", linewidth=2.5)
        axs[3].set_ylabel("BR/MIN")
        axs[3].legend(loc="upper right")
        axs[3].grid(True, linestyle="--", alpha=0.5)
        
        # 5. SpO2
        axs[4].plot(timestamps, spo2_vals, color='#00cc66', label="SpO2 (%)", linewidth=2.5)
        axs[4].set_ylabel("SpO2 %")
        axs[4].set_xlabel("Time (seconds)")
        axs[4].legend(loc="upper right")
        axs[4].grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        
        # Save the finalized stacked plot
        plot_path = os.path.join(participant_folder, "vitals_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Plotting error: {e}")
# To call in main:
# from data_logging.visualization import generate_session_plot

import matplotlib.pyplot as plt
import os
import csv
import numpy as np

# A fallback just in case this isn't passed from main
MATPLOTLIB_AVAILABLE = True

def generate_session_plot(participant_folder, raw_csv_path):
    if not MATPLOTLIB_AVAILABLE:
        print("[Plot Error] Matplotlib is not available on this system.")
        return
    if not os.path.exists(raw_csv_path):
        print(f"[Plot Error] Target CSV file not found: {raw_csv_path}")
        return

    try:
        timestamps = []
        hr_vals, rr_vals, spo2_vals, rppg_vals = [], [], [], []
        r_vals, g_vals, b_vals = [], [], []
        
        with open(raw_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("ProtocolPhase") == "ACQUISITION":
                    try:
                        t = float(row.get("PhaseElapsed", 0))
                        
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
        
        if len(timestamps) < 5:
            print(f"[Plot Warning] Not enough valid acquisition rows ({len(timestamps)}) found in {raw_csv_path}")
            return
        
        # Build 5-panel stacked chart
        fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        fig.suptitle("Participant Session Summary", fontsize=16, fontweight="bold")
        
        # 1. RGB Intensity
        axs[0].plot(timestamps, r_vals, color='#f472b6', label="Mean Red")
        axs[0].plot(timestamps, g_vals, color='#4ade80', label="Mean Green")
        axs[0].plot(timestamps, b_vals, color='#60a5fa', label="Mean Blue")
        axs[0].set_ylabel("RGB Intensity")
        axs[0].legend(loc="upper right")
        
        # 2. Filtered rPPG
        axs[1].plot(timestamps, rppg_vals, color='purple', label="Filtered rPPG Waveform")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend(loc="upper right")
        
        # 3. Heart Rate
        axs[2].plot(timestamps, hr_vals, color='#ef4444', label="Heart Rate (BPM)", linewidth=2)
        axs[2].set_ylabel("BPM")
        axs[2].legend(loc="upper right")
        
        # 4. Respiratory Rate
        axs[3].plot(timestamps, rr_vals, color='#3b82f6', label="Respiratory Rate (BR/MIN)", linewidth=2)
        axs[3].set_ylabel("BR/MIN")
        axs[3].legend(loc="upper right")
        
        # 5. SpO2
        axs[4].plot(timestamps, spo2_vals, color='#10b981', label="SpO2 (%)", linewidth=2)
        axs[4].set_ylabel("SpO2 %")
        axs[4].set_xlabel("Time (seconds)")
        axs[4].legend(loc="upper right")
        
        for ax in axs:
            ax.grid(True, linestyle="--", alpha=0.5)
            
        plt.tight_layout()
        
        # Save as PNG & PDF vector graphic
        png_path = os.path.join(participant_folder, "vitals_plot.png")
        pdf_path = os.path.join(participant_folder, "vitals_plot.pdf")
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        
        print(f"[Plot Success] Saved plots to:\n -> {png_path}\n -> {pdf_path}")
        
    except Exception as e:
        print(f"[Plot Error] Plotting failed with exception: {e}")
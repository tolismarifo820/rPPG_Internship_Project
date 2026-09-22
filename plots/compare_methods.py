import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Updated paths to look one level up into the root project directory
csv_path = "../logs/offline_evaluation_results.csv"
output_image_path = "../logs/rppg_performance_comparison.png"

# Verify the evaluation file exists before attempting to plot
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}. Make sure you are running this from the root rPPG_project folder.")
    exit()

# Load the batch evaluation data
df = pd.read_csv(csv_path)

# Set visualization styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Spectral SNR by Method and EVM
sns.barplot(data=df, x="method", y="snr", hue="evm", ax=axes[0, 0], palette="Blues_d", errorbar=("ci", 68))
axes[0, 0].set_title("Spectral Signal-to-Noise Ratio (SNR) by Method & EVM (Higher is Better)", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("rPPG Extraction Method")
axes[0, 0].set_ylabel("SNR (dB)")
axes[0, 0].legend(title="EVM Mode")

# 2. CWT Ridge Variance by Method and ROI
sns.barplot(data=df, x="method", y="ridge_variance", hue="roi", ax=axes[0, 1], palette="Greens_d", errorbar=("ci", 68))
axes[0, 1].set_title("CWT Ridge Variance by Method & ROI (Lower is Better)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("rPPG Extraction Method")
axes[0, 1].set_ylabel("Ridge Variance (Scale units)")
axes[0, 1].legend(title="ROI")

# 3. Temporal Heart Rate Variance
sns.boxplot(data=df, x="method", y="temporal_variance", ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("Temporal Heart Rate Variance across Sliding Windows (Lower is Better)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("rPPG Extraction Method")
axes[1, 0].set_ylabel("BPM Std Dev")

# 4. Pipeline Configuration Heatmap
# Combine ROI and EVM into a single string for the heatmap x-axis
df["pipeline"] = df["roi"] + " + " + df["evm"]
pivot_snr = df.pivot_table(index="method", columns="pipeline", values="snr", aggfunc="mean")

sns.heatmap(pivot_snr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Mean SNR (dB)"}, ax=axes[1, 1])
axes[1, 1].set_title("Mean SNR Matrix: Method vs ROI & EVM Pipeline", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Pipeline (ROI + EVM)")
axes[1, 1].set_ylabel("Method")

# Final layout adjustments and save
plt.tight_layout()
plt.savefig(output_image_path, dpi=300)
print(f"Dashboard saved successfully to {output_image_path}")
plt.show()
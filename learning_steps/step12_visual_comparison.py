# step12_visual_comparison.py
# Purpose: labeled visual comparison (Original, Color Corrected,
# Contrast Enhanced, Final) plus metric graphs.

import cv2
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

from dip_functions import (resize_image, white_balance, auto_gamma_correction,
                           apply_clahe, unsharp_mask, all_metrics)

# ---------- 1. Paths ----------
base_folder    = os.path.dirname(os.path.abspath(__file__))
input_path     = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")
output_folder  = os.path.join(base_folder, "..", "output_images")
results_folder = os.path.join(base_folder, "..", "results")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# ---------- 2. Settings ----------
MAX_WIDTH, MAX_GAIN, TARGET_MEAN = 800, 3.0, 110
CLIP_LIMIT, TILE_GRID, SIGMA, AMOUNT = 2.0, (8, 8), 1.5, 0.8

# ---------- 3. Read ----------
img = cv2.imread(input_path)
if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()
img = resize_image(img, MAX_WIDTH)

# ---------- 4. Build the four stages ----------
color_corrected   = white_balance(img, MAX_GAIN)
contrast_enhanced = apply_clahe(auto_gamma_correction(color_corrected, TARGET_MEAN),
                                CLIP_LIMIT, TILE_GRID)
final_enhanced    = unsharp_mask(contrast_enhanced, SIGMA, AMOUNT)

cv2.imwrite(os.path.join(output_folder, "step12_final_enhanced.jpg"), final_enhanced)

stages = [
    ("1. Original",                              img),
    ("2. Color Corrected\n(White Balance)",      color_corrected),
    ("3. Contrast Enhanced\n(WB + Gamma + CLAHE)", contrast_enhanced),
    ("4. Final Enhanced\n(+ Sharpening)",        final_enhanced),
]
short_names = ["Original", "Color Corrected", "Contrast Enhanced", "Final"]

# ---------- 5. Metrics for every stage ----------
metric_names = ["Brightness", "Contrast", "Entropy", "UCIQE", "UIQM"]
values = [all_metrics(image) for _, image in stages]     # list of 4 tuples

print(f"{'Stage':<20}" + "".join(f"{m:>12}" for m in metric_names))
for name, v in zip(short_names, values):
    print(f"{name:<20}" + "".join(f"{x:>12.3f}" for x in v))

with open(os.path.join(results_folder, "step12_metrics_table.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Stage"] + metric_names)
    for name, v in zip(short_names, values):
        writer.writerow([name] + [round(x, 4) for x in v])

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 6. FIGURE 1: labeled visual comparison ----------
fig1, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, (title, image), v in zip(axes, stages, values):
    ax.imshow(to_rgb(image))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.5, -0.04, f"UCIQE {v[3]:.3f}  |  UIQM {v[4]:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.axis("off")
plt.tight_layout()
fig1.savefig(os.path.join(results_folder, "step12_visual_comparison.png"),
             dpi=200, bbox_inches="tight")

# ---------- 7. FIGURE 2: RGB histograms of the four stages ----------
fig2, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for ax, (title, image) in zip(axes, stages):
    for ch_index, color in enumerate(["blue", "green", "red"]):
        hist = cv2.calcHist([image], [ch_index], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_title(title.split("\n")[0], fontsize=10)
    ax.set_xlim([0, 255])
    ax.set_xlabel("Pixel value")
plt.tight_layout()
fig2.savefig(os.path.join(results_folder, "step12_histograms.png"), dpi=150)

# ---------- 8. FIGURE 3: one bar chart per metric ----------
colors = ["gray", "tab:blue", "tab:green", "tab:orange"]
fig3, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, (ax, m) in enumerate(zip(axes, metric_names)):
    vals = [v[i] for v in values]
    bars = ax.bar(short_names, vals, color=colors)
    ax.set_title(m)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    for bar, x in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, x, f"{x:.2f}",
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig3.savefig(os.path.join(results_folder, "step12_metric_bars.png"), dpi=150)

# ---------- 9. FIGURE 4: percent change compared with the Original ----------
original = np.array(values[0])
fig4 = plt.figure(figsize=(9, 5))
width = 0.25
x = np.arange(len(metric_names))
for j in range(1, 4):                                   # the three enhanced stages
    change = (np.array(values[j]) / original - 1) * 100
    plt.bar(x + (j - 2) * width, change, width, label=short_names[j], color=colors[j])
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(x, metric_names)
plt.ylabel("Change compared with Original (%)")
plt.title("Improvement over the original image")
plt.legend()
plt.tight_layout()
fig4.savefig(os.path.join(results_folder, "step12_percent_change.png"), dpi=150)

plt.show()
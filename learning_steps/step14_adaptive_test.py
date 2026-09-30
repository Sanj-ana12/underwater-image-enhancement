# step14_adaptive_test.py
# Purpose: compare the FIXED pipeline with the ADAPTIVE pipeline.
# Uses every image in input_images/ plus a dark and a hazy copy of the first one.

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pipeline import load_image, enhance_image, calculate_metrics
from adaptive import enhance_adaptive
from dip_functions import gamma_correction

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")
MAX_IMAGES  = 5                       # limit real images so the figure stays readable

base_folder    = os.path.dirname(os.path.abspath(__file__))
input_folder   = os.path.join(base_folder, "..", "input_images")
results_folder = os.path.join(base_folder, "..", "results")
os.makedirs(results_folder, exist_ok=True)

# ---------- 1. Collect test cases: (name, image) ----------
files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(IMAGE_TYPES))
cases = [(os.path.splitext(f)[0], load_image(os.path.join(input_folder, f)))
         for f in files[:MAX_IMAGES]]
if not cases:
    raise SystemExit("No images found in input_images/")

# Extra cases made from the first image
first_name, first_img = cases[0]

dark = gamma_correction(first_img, 2.2)                        # gamma > 1 darkens
veil = np.array([190, 160, 60], dtype=np.float32)              # blue-green haze (B, G, R)
hazy = np.clip(first_img.astype(np.float32) * 0.4 + veil * 0.6, 0, 255).astype(np.uint8)

cases.append((f"{first_name} (made dark)", dark))
cases.append((f"{first_name} (made hazy)", hazy))

# ---------- 2. Run both pipelines on every case ----------
metric_names = ["Brightness", "Contrast", "Entropy", "UCIQE", "UIQM"]
csv_rows = []
results = []                                                   # (name, original, fixed, adaptive)

for name, image in cases:
    fixed, _ = enhance_image(image)
    adaptive, _, decisions = enhance_adaptive(image)
    results.append((name, image, fixed, adaptive))

    print(f"\n=================== {name} ===================")
    print("Adaptive decisions:")
    for line in decisions:
        print("  -", line)

    print(f"\n{'':<10}" + "".join(f"{m:>12}" for m in metric_names))
    for label, img_version in [("Original", image), ("Fixed", fixed), ("Adaptive", adaptive)]:
        m = calculate_metrics(img_version)
        print(f"{label:<10}" + "".join(f"{m[k]:>12.3f}" for k in metric_names))
        csv_rows.append([name, label] + [round(m[k], 4) for k in metric_names])

with open(os.path.join(results_folder, "step14_adaptive_vs_fixed.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Version"] + metric_names)
    writer.writerows(csv_rows)

# ---------- 3. Figure: one row per case ----------
n = len(results)
fig, axes = plt.subplots(n, 3, figsize=(15, 3.8 * n), squeeze=False)
for row, (name, original, fixed, adaptive) in enumerate(results):
    for col, (title, image) in enumerate([("Original", original),
                                          ("Fixed pipeline", fixed),
                                          ("Adaptive pipeline", adaptive)]):
        axes[row][col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row][col].set_title(f"{name}\n{title}" if col == 0 else title, fontsize=10)
        axes[row][col].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(results_folder, "step14_adaptive_vs_fixed.png"), dpi=150)
print("\nSaved figure and CSV in:", os.path.abspath(results_folder))

plt.show()
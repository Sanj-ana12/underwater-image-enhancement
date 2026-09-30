# step4_degradation_demo.py
# Purpose: simulate underwater degradation (color loss + haze)
#          on a normal photo, at different distances.

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Paths ----------
base_folder    = os.path.dirname(os.path.abspath(__file__))
normal_path    = os.path.join(base_folder, "..", "input_images", "normal1.jpg")
fallback_path  = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")
results_folder = os.path.join(base_folder, "..", "results")
os.makedirs(results_folder, exist_ok=True)

# ---------- 2. Read the image (normal photo preferred) ----------
img = cv2.imread(normal_path)
if img is None:
    print("normal1.jpg not found. Using underwater1.jpg instead.")
    img = cv2.imread(fallback_path)
if img is None:
    print("ERROR: no image found in input_images/")
    sys.exit()

# Shrink big images so the demo runs fast
img = cv2.resize(img, (500, int(500 * img.shape[0] / img.shape[1])))

# ---------- 3. Model settings (order is B, G, R like OpenCV) ----------
# Attenuation coefficient per color: how quickly each color is absorbed.
# Red is absorbed fastest. These values are illustrative, not exact.
beta = np.array([0.05, 0.12, 0.60])

# Veiling light: the blue-green glow from backscatter (B, G, R)
veil = np.array([190.0, 160.0, 60.0])

# ---------- 4. Function that damages the image ----------
def simulate_underwater(image, distance):
    """Return a degraded copy of 'image' as if seen through 'distance' meters of water."""
    original = image.astype(np.float32)                 # use decimals for math
    t = np.exp(-beta * distance)                        # transmission of each channel
    degraded = original * t + veil * (1 - t)            # absorption + backscatter
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)  # back to 0-255 integers
    return degraded

# ---------- 5. Create three versions ----------
distances = [1, 3, 6]                                   # meters
versions  = [img] + [simulate_underwater(img, d) for d in distances]
titles    = ["Original (no water)"] + [f"{d} m of water" for d in distances]

# ---------- 6. Print channel means for each version ----------
print("Distance   Blue mean   Green mean   Red mean   Gray std (contrast)")
for title, v in zip(titles, versions):
    b, g, r = cv2.split(v)
    gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    print(f"{title:<20} {b.mean():7.1f} {g.mean():11.1f} {r.mean():10.1f} {gray.std():14.1f}")

# ---------- 7. Show images and histograms ----------
plt.figure(figsize=(14, 6))
for i, (title, v) in enumerate(zip(titles, versions)):
    # Top row: the images
    plt.subplot(2, 4, i + 1)
    plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

    # Bottom row: color histograms
    plt.subplot(2, 4, i + 5)
    for ch_index, color in enumerate(["blue", "green", "red"]):
        hist = cv2.calcHist([v], [ch_index], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 255])
    plt.xlabel("Pixel value")

plt.tight_layout()
save_path = os.path.join(results_folder, "step4_degradation_demo.png")
plt.savefig(save_path, dpi=150)
print("Saved:", os.path.abspath(save_path))
plt.show()
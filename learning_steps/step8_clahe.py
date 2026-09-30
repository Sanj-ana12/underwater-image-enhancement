# step8_clahe.py
# Purpose: improve contrast using
#   (1) normal histogram equalization  and  (2) CLAHE.
# Both are applied on the L (lightness) channel of the LAB color space.
# We also experiment with clipLimit and tileGridSize.

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Paths ----------
base_folder    = os.path.dirname(os.path.abspath(__file__))
input_path     = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")
output_folder  = os.path.join(base_folder, "..", "output_images")
results_folder = os.path.join(base_folder, "..", "results")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# ---------- 2. Settings you can change ----------
MAX_WIDTH  = 800
CLIP_LIMIT = 2.0        # default CLAHE strength
TILE_GRID  = (8, 8)     # default number of tiles (columns, rows)

# ---------- 3. Helper functions ----------
def resize_image(image, max_width):
    """Shrink image so width <= max_width. Keeps aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def histogram_equalization(image):
    """Normal (global) histogram equalization on the L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)      # BGR -> LAB
    l, a, b = cv2.split(lab)                          # separate lightness and color
    l_eq = cv2.equalizeHist(l)                        # equalize only lightness
    lab_eq = cv2.merge((l_eq, a, b))                  # put the channels back
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)    # LAB -> BGR

def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    """
    CLAHE on the L channel.
    image      : BGR image (uint8)
    clip_limit : contrast limit (higher = stronger)
    tile_grid  : (columns, rows) of tiles
    Returns the enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def brightness_and_contrast(image):
    """Return (mean brightness, contrast) from the grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean(), gray.std()

def to_rgb(image):
    """BGR -> RGB for Matplotlib."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 4. Read and resize ----------
img = cv2.imread(input_path)
if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()

img = resize_image(img, MAX_WIDTH)

# ---------- 5. Main comparison: Original vs HE vs CLAHE ----------
he_img    = histogram_equalization(img)
clahe_img = apply_clahe(img, CLIP_LIMIT, TILE_GRID)

main_versions = [("Original", img),
                 ("Histogram Equalization", he_img),
                 (f"CLAHE (clip {CLIP_LIMIT}, tiles {TILE_GRID[0]}x{TILE_GRID[1]})", clahe_img)]

print(f"{'Version':<40}{'Brightness':>12}{'Contrast':>12}")
for name, image in main_versions:
    bright, contrast = brightness_and_contrast(image)
    print(f"{name:<40}{bright:>12.1f}{contrast:>12.1f}")

cv2.imwrite(os.path.join(output_folder, "step8_hist_eq.jpg"), he_img)
cv2.imwrite(os.path.join(output_folder, "step8_clahe.jpg"),   clahe_img)

# ---------- 6. Figure 1: images (top) and grayscale histograms (bottom) ----------
fig1 = plt.figure(figsize=(14, 7))
for i, (name, image) in enumerate(main_versions):
    plt.subplot(2, 3, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name, fontsize=10)
    plt.axis("off")

    plt.subplot(2, 3, i + 4)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist, color="black")
    plt.xlim([0, 255])
    plt.xlabel("Pixel value")
    plt.ylabel("Pixels")
plt.tight_layout()
fig1.savefig(os.path.join(results_folder, "step8_clahe_comparison.png"), dpi=150)

# ---------- 7. Experiment: change clipLimit, then tileGridSize ----------
clip_values = [1.0, 2.0, 4.0, 8.0]
tile_values = [(2, 2), (4, 4), (8, 8), (16, 16)]

experiments = [(f"clip = {c}, tiles 8x8", apply_clahe(img, c, (8, 8))) for c in clip_values]
experiments += [(f"clip = 2.0, tiles {t[0]}x{t[1]}", apply_clahe(img, 2.0, t)) for t in tile_values]

print(f"\n{'Experiment':<32}{'Brightness':>12}{'Contrast':>12}")
for name, image in experiments:
    bright, contrast = brightness_and_contrast(image)
    print(f"{name:<32}{bright:>12.1f}{contrast:>12.1f}")

fig2 = plt.figure(figsize=(16, 8))
for i, (name, image) in enumerate(experiments):
    plt.subplot(2, 4, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name, fontsize=10)
    plt.axis("off")
plt.tight_layout()
fig2.savefig(os.path.join(results_folder, "step8_parameter_experiment.png"), dpi=150)

plt.show()
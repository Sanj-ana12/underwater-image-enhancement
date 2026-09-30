# step9_sharpening.py
# Purpose: sharpen an underwater image using unsharp masking.
# Part A: show every stage (Original -> Blurred -> Detail -> Sharpened).
# Part B: experiment with the sigma and amount parameters.

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
MAX_WIDTH = 800
SIGMA     = 1.5     # blur size used to find the details
AMOUNT    = 1.0     # how strongly details are added back

# ---------- 3. Helper functions ----------
def resize_image(image, max_width):
    """Shrink image so width <= max_width. Keeps aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def unsharp_mask(image, sigma=1.5, amount=1.0):
    """
    Sharpen an image with unsharp masking.
    image  : BGR image (uint8)
    sigma  : size of the blur (bigger = boosts larger details)
    amount : strength of sharpening (0 = no change)
    Returns the sharpened image.
    """
    # (0, 0) means: work out the kernel size from sigma automatically
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # sharpened = (1 + amount) * image - amount * blurred
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

def sharpness_score(image):
    """Variance of the Laplacian: higher = more edge energy (sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

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

# ---------- 5. PART A: show every stage ----------
blurred   = cv2.GaussianBlur(img, (0, 0), SIGMA)
sharpened = unsharp_mask(img, SIGMA, AMOUNT)

# Detail layer = original - blurred. It can be negative, so we use decimals,
# multiply by 4 to make it visible, and shift by 128 so "no detail" shows as gray.
detail = img.astype(np.float32) - blurred.astype(np.float32)
detail_view = np.clip(detail * 4 + 128, 0, 255).astype(np.uint8)

cv2.imwrite(os.path.join(output_folder, "step9_sharpened.jpg"), sharpened)

print(f"Settings: sigma = {SIGMA}, amount = {AMOUNT}")
print(f"Sharpness score  Original : {sharpness_score(img):8.1f}")
print(f"Sharpness score  Blurred  : {sharpness_score(blurred):8.1f}")
print(f"Sharpness score  Sharpened: {sharpness_score(sharpened):8.1f}")

fig1 = plt.figure(figsize=(16, 4.5))
stages = [(img, "1. Original"),
          (blurred, f"2. Blurred (sigma {SIGMA})"),
          (detail_view, "3. Detail layer (x4, gray = 0)"),
          (sharpened, f"4. Sharpened (amount {AMOUNT})")]
for i, (image, title) in enumerate(stages):
    plt.subplot(1, 4, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(title, fontsize=10)
    plt.axis("off")
plt.tight_layout()
fig1.savefig(os.path.join(results_folder, "step9_sharpening_stages.png"), dpi=150)

# ---------- 6. PART B: parameter experiments ----------
amounts = [0.5, 1.0, 2.0, 3.0]
sigmas  = [0.8, 1.5, 3.0, 5.0]

experiments  = [(f"amount = {a}, sigma 1.5", unsharp_mask(img, 1.5, a)) for a in amounts]
experiments += [(f"amount = 1.0, sigma {s}", unsharp_mask(img, s, 1.0)) for s in sigmas]

print(f"\n{'Experiment':<28}{'Sharpness score':>18}")
print(f"{'Original':<28}{sharpness_score(img):>18.1f}")
for name, image in experiments:
    print(f"{name:<28}{sharpness_score(image):>18.1f}")

fig2 = plt.figure(figsize=(16, 8))
for i, (name, image) in enumerate(experiments):
    plt.subplot(2, 4, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name, fontsize=10)
    plt.axis("off")
plt.tight_layout()
fig2.savefig(os.path.join(results_folder, "step9_parameter_experiment.png"), dpi=150)

plt.show()
# step5_preprocessing.py
# Purpose: resize the image and compare Gaussian vs Median filtering.
# Part A: on your real image (Original -> Resized -> Gaussian -> Median)
# Part B: on artificially noisy copies, to see when each filter is useful.

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
MAX_WIDTH = 800     # images wider than this will be shrunk
KSIZE     = 5       # kernel size for filters (must be an odd number: 3, 5, 7...)

# ---------- 3. Read the image ----------
img = cv2.imread(input_path)
if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()

# ---------- 4. Helper functions ----------
def resize_image(image, max_width):
    """Shrink 'image' so its width is at most max_width. Keeps the aspect ratio."""
    h, w = image.shape[:2]                       # first two values of shape
    if w <= max_width:
        return image.copy()                      # already small enough
    scale = max_width / w                        # for example 800 / 4000 = 0.2
    new_size = (max_width, int(h * scale))       # OpenCV wants (width, height)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def mean_abs_diff(a, b):
    """Average absolute difference between two images (0 = identical)."""
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))

def to_rgb(image):
    """Convert BGR to RGB so Matplotlib shows correct colors."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 5. PART A: resize and filter the real image ----------
resized  = resize_image(img, MAX_WIDTH)
gaussian = cv2.GaussianBlur(resized, (KSIZE, KSIZE), 0)
median   = cv2.medianBlur(resized, KSIZE)

print("Original size :", img.shape[1], "x", img.shape[0])
print("Resized size  :", resized.shape[1], "x", resized.shape[0])
print(f"\nChange caused by Gaussian filter (mean abs diff): {mean_abs_diff(resized, gaussian):.2f}")
print(f"Change caused by Median filter   (mean abs diff): {mean_abs_diff(resized, median):.2f}")

# Save the resized image (used in later steps)
cv2.imwrite(os.path.join(output_folder, "step5_resized.jpg"), resized)

fig1 = plt.figure(figsize=(14, 4))
panels = [(img, "Original"), (resized, "Resized"),
          (gaussian, f"Gaussian {KSIZE}x{KSIZE}"), (median, f"Median {KSIZE}")]
for i, (image, title) in enumerate(panels):
    plt.subplot(1, 4, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
fig1.savefig(os.path.join(results_folder, "step5_partA_preprocessing.png"), dpi=150)

# ---------- 6. PART B: add fake noise, then filter it ----------
rng = np.random.default_rng(42)                  # fixed seed = same result every run

# (i) Salt-and-pepper noise: about 1% black dots and 1% white dots
sp_noisy = resized.copy()
rand_map = rng.random(resized.shape[:2])         # one random number per pixel
sp_noisy[rand_map < 0.01] = 0                    # black dots
sp_noisy[rand_map > 0.99] = 255                  # white dots

# (ii) Gaussian noise: small random change added to every pixel
noise    = rng.normal(0, 20, resized.shape)      # mean 0, spread 20
g_noisy  = np.clip(resized.astype(np.float32) + noise, 0, 255).astype(np.uint8)

rows = [("Salt-and-pepper noise", sp_noisy), ("Gaussian noise", g_noisy)]

fig2 = plt.figure(figsize=(14, 8))
print("\nError compared with the clean resized image (lower = closer to clean):")
for r, (name, noisy) in enumerate(rows):
    g_fixed = cv2.GaussianBlur(noisy, (KSIZE, KSIZE), 0)
    m_fixed = cv2.medianBlur(noisy, KSIZE)
    print(f"\n{name}")
    print(f"  Noisy    : {mean_abs_diff(resized, noisy):.2f}")
    print(f"  Gaussian : {mean_abs_diff(resized, g_fixed):.2f}")
    print(f"  Median   : {mean_abs_diff(resized, m_fixed):.2f}")

    for c, (image, title) in enumerate([(noisy, name),
                                        (g_fixed, "After Gaussian"),
                                        (m_fixed, "After Median")]):
        plt.subplot(2, 3, r * 3 + c + 1)
        plt.imshow(to_rgb(image))
        plt.title(title)
        plt.axis("off")
plt.tight_layout()
fig2.savefig(os.path.join(results_folder, "step5_partB_noise_demo.png"), dpi=150)

plt.show()      # opens both figures
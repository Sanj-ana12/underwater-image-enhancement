# step7_gamma_correction.py
# Purpose: brighten a dark underwater image with gamma correction.
# We try several gamma values and also calculate an automatic gamma.

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
MAX_WIDTH    = 800
GAMMA_VALUES = [0.5, 0.7, 1.0, 1.5]    # gammas to compare
TARGET_MEAN  = 110                      # brightness we want for the automatic gamma

# ---------- 3. Helper functions ----------
def resize_image(image, max_width):
    """Shrink image so width <= max_width. Keeps aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def gamma_correction(image, gamma):
    """
    Apply gamma correction.
    image : BGR image (uint8)
    gamma : number > 0.  gamma < 1 brightens, gamma > 1 darkens.
    Returns the corrected image.
    """
    # Build the lookup table: 256 values, one for each possible pixel value
    table = np.array([255 * (i / 255.0) ** gamma for i in range(256)])
    table = np.clip(np.round(table), 0, 255).astype(np.uint8)
    # Replace every pixel by its table entry
    return cv2.LUT(image, table)

def auto_gamma(image, target_mean=110, low=0.4, high=1.0):
    """Calculate a gamma that moves the mean brightness towards target_mean."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.clip(gray.mean(), 1, 254)              # avoid log(0) and log(1)
    gamma = np.log(target_mean / 255.0) / np.log(mean / 255.0)
    return float(np.clip(gamma, low, high))          # keep gamma in a safe range

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

# ---------- 5. Apply the fixed gamma values ----------
versions = []
for g in GAMMA_VALUES:
    label = "Original (gamma 1.0)" if g == 1.0 else f"gamma = {g}"
    versions.append((label, gamma_correction(img, g)))

# ---------- 6. Apply the automatic gamma ----------
g_auto = auto_gamma(img, TARGET_MEAN)
versions.append((f"auto gamma = {g_auto:.2f}", gamma_correction(img, g_auto)))

# ---------- 7. Print brightness and contrast for each version ----------
print(f"Automatic gamma chosen: {g_auto:.2f} (target mean brightness = {TARGET_MEAN})\n")
print(f"{'Version':<24}{'Mean brightness':>18}{'Contrast (std)':>18}")
for label, image in versions:
    bright, contrast = brightness_and_contrast(image)
    print(f"{label:<24}{bright:>18.1f}{contrast:>18.1f}")

# Save the automatic result for later steps
cv2.imwrite(os.path.join(output_folder, "step7_gamma_auto.jpg"), versions[-1][1])

# ---------- 8. Figure 1: the gamma curves ----------
fig1 = plt.figure(figsize=(6, 5))
x = np.linspace(0, 1, 256)                         # input values from 0 to 1
for g in [0.5, 0.7, 1.0, 1.5, 2.0]:
    plt.plot(x * 255, (x ** g) * 255, label=f"gamma = {g}")
plt.xlabel("Input pixel value")
plt.ylabel("Output pixel value")
plt.title("Gamma curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig1.savefig(os.path.join(results_folder, "step7_gamma_curves.png"), dpi=150)

# ---------- 9. Figure 2: images (top) and grayscale histograms (bottom) ----------
n = len(versions)
fig2 = plt.figure(figsize=(16, 6))
for i, (label, image) in enumerate(versions):
    plt.subplot(2, n, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(label, fontsize=10)
    plt.axis("off")

    plt.subplot(2, n, n + i + 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist, color="black")
    plt.xlim([0, 255])
    plt.xlabel("Pixel value")
plt.tight_layout()
fig2.savefig(os.path.join(results_folder, "step7_gamma_comparison.png"), dpi=150)

plt.show()
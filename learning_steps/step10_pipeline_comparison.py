# step10_pipeline_comparison.py
# Purpose: combine white balance, gamma, CLAHE and sharpening into pipelines,
# then compare different combinations and different orders.

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
MAX_WIDTH   = 800
MAX_GAIN    = 3.0        # white balance gain limit
TARGET_MEAN = 110        # brightness target for automatic gamma
CLIP_LIMIT  = 2.0        # CLAHE strength
TILE_GRID   = (8, 8)     # CLAHE tiles
SIGMA       = 1.5        # sharpening blur size
AMOUNT      = 0.8        # sharpening strength

# ---------- 3. Building-block functions (from Steps 5 to 9) ----------
def resize_image(image, max_width):
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def white_balance(image, max_gain=3.0):
    """Gray World white balance with a gain limit. Returns the corrected image."""
    img_f = image.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    gains = np.minimum(means.mean() / means, max_gain)
    return np.clip(img_f * gains, 0, 255).astype(np.uint8)

def auto_gamma(image, target_mean=110, low=0.4, high=1.0):
    """Gamma that moves the mean brightness towards target_mean."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.clip(gray.mean(), 1, 254)
    gamma = np.log(target_mean / 255.0) / np.log(mean / 255.0)
    return float(np.clip(gamma, low, high))

def gamma_correction(image, gamma):
    """Gamma correction with a lookup table. gamma < 1 brightens."""
    table = np.array([255 * (i / 255.0) ** gamma for i in range(256)])
    table = np.clip(np.round(table), 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)

def auto_gamma_correction(image, target_mean=110):
    """Find the gamma from the image itself, then apply it."""
    return gamma_correction(image, auto_gamma(image, target_mean))

def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE on the L channel of LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

def unsharp_mask(image, sigma=1.5, amount=1.0):
    """Unsharp masking."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

# ---------- 4. Short names for each step (so pipelines are easy to write) ----------
def step_wb(x):      return white_balance(x, MAX_GAIN)
def step_gamma(x):   return auto_gamma_correction(x, TARGET_MEAN)
def step_clahe(x):   return apply_clahe(x, CLIP_LIMIT, TILE_GRID)
def step_sharpen(x): return unsharp_mask(x, SIGMA, AMOUNT)

def run_pipeline(image, steps):
    """
    Apply a list of steps in order.
    image : starting BGR image
    steps : list of functions, e.g. [step_wb, step_clahe]
    Returns the final image.
    """
    result = image
    for step in steps:
        result = step(result)      # output of one step becomes input of the next
    return result

# ---------- 5. Measurement helpers ----------
def measure(image):
    """Return (brightness, contrast, sharpness, R/B ratio)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    b, g, r = image.reshape(-1, 3).astype(np.float32).mean(axis=0)
    return gray.mean(), gray.std(), sharp, r / b

def print_table(title, versions):
    print(f"\n{title}")
    print(f"{'Method':<34}{'Bright':>8}{'Contrast':>10}{'Sharp':>9}{'R/B':>7}")
    for name, image in versions:
        br, co, sh, rb = measure(image)
        print(f"{name:<34}{br:>8.1f}{co:>10.1f}{sh:>9.1f}{rb:>7.2f}")

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 6. Read and resize ----------
img = cv2.imread(input_path)
if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()
img = resize_image(img, MAX_WIDTH)

# ---------- 7. PART A: Methods A, B, C ----------
method_a = run_pipeline(img, [step_wb, step_clahe])
method_b = run_pipeline(img, [step_wb, step_gamma, step_clahe])
method_c = run_pipeline(img, [step_wb, step_gamma, step_clahe, step_sharpen])

part_a = [("Original", img),
          ("A: WB > CLAHE", method_a),
          ("B: WB > Gamma > CLAHE", method_b),
          ("C: WB > Gamma > CLAHE > Sharpen", method_c)]

print_table("PART A: adding one technique at a time", part_a)

# Show which gamma was chosen inside Method B
print("\nAutomatic gamma used in B and C:", round(auto_gamma(step_wb(img), TARGET_MEAN), 2))

cv2.imwrite(os.path.join(output_folder, "step10_method_A.jpg"), method_a)
cv2.imwrite(os.path.join(output_folder, "step10_method_B.jpg"), method_b)
cv2.imwrite(os.path.join(output_folder, "step10_method_C.jpg"), method_c)

fig1 = plt.figure(figsize=(16, 7))
for i, (name, image) in enumerate(part_a):
    plt.subplot(2, 4, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name, fontsize=9)
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    for ch_index, color in enumerate(["blue", "green", "red"]):
        hist = cv2.calcHist([image], [ch_index], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 255])
    plt.xlabel("Pixel value")
plt.tight_layout()
fig1.savefig(os.path.join(results_folder, "step10_partA_methods.png"), dpi=150)

# ---------- 8. PART B: does the order matter? ----------
variants = [
    ("Original",                        img),
    ("C: WB > Gamma > CLAHE > Sharpen", method_c),
    ("D: CLAHE > WB",                   run_pipeline(img, [step_clahe, step_wb])),
    ("E: Gamma > WB > CLAHE",           run_pipeline(img, [step_gamma, step_wb, step_clahe])),
    ("F: WB > Sharpen > CLAHE",         run_pipeline(img, [step_wb, step_sharpen, step_clahe])),
    ("G: WB > CLAHE > Gamma",           run_pipeline(img, [step_wb, step_clahe, step_gamma])),
]

print_table("PART B: does the order matter?", variants)

fig2 = plt.figure(figsize=(15, 8))
for i, (name, image) in enumerate(variants):
    plt.subplot(2, 3, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name, fontsize=10)
    plt.axis("off")
plt.tight_layout()
fig2.savefig(os.path.join(results_folder, "step10_partB_order.png"), dpi=150)

plt.show()
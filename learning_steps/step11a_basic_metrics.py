# step11a_basic_metrics.py
# Purpose: measure brightness, contrast and entropy for Original and
# Methods A, B, C, then show a comparison table and bar charts.

import cv2
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Paths ----------
base_folder    = os.path.dirname(os.path.abspath(__file__))
input_path     = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")
results_folder = os.path.join(base_folder, "..", "results")
os.makedirs(results_folder, exist_ok=True)

# ---------- 2. Settings ----------
MAX_WIDTH   = 800
MAX_GAIN    = 3.0
TARGET_MEAN = 110
CLIP_LIMIT  = 2.0
TILE_GRID   = (8, 8)
SIGMA       = 1.5
AMOUNT      = 0.8

# ---------- 3. Enhancement functions (same as Step 10) ----------
def resize_image(image, max_width):
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    return cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)

def white_balance(image, max_gain=3.0):
    img_f = image.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    gains = np.minimum(means.mean() / means, max_gain)
    return np.clip(img_f * gains, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma):
    table = np.array([255 * (i / 255.0) ** gamma for i in range(256)])
    table = np.clip(np.round(table), 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)

def auto_gamma_correction(image, target_mean=110, low=0.4, high=1.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.clip(gray.mean(), 1, 254)
    gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(mean / 255.0), low, high))
    return gamma_correction(image, gamma)

def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

def unsharp_mask(image, sigma=1.5, amount=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

# ---------- 4. THE NEW PART: the three metrics ----------
def mean_brightness(image):
    """Average grayscale value (0-255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())

def contrast_std(image):
    """Standard deviation of the grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())

def entropy(image):
    """Shannon entropy of the grayscale histogram (0 to 8 bits)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()  # 256 counts
    p = hist / hist.sum()          # counts -> probabilities
    p = p[p > 0]                   # drop zeros (log2(0) is undefined)
    return float(-np.sum(p * np.log2(p)))

def basic_metrics(image):
    """Return (brightness, contrast, entropy) for one image."""
    return mean_brightness(image), contrast_std(image), entropy(image)

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 5. Sanity test: do the metrics behave as the theory says? ----------
print("SANITY TEST (synthetic images, 200 x 200)")
flat  = np.full((200, 200, 3), 128, dtype=np.uint8)                         # all pixels equal
noise = np.random.default_rng(1).integers(0, 256, (200, 200, 3)).astype(np.uint8)  # random pixels
print(f"  Flat gray image : entropy = {entropy(flat):.2f}   (expected 0)")
print(f"  Random noise    : entropy = {entropy(noise):.2f}   (expected close to 8)")
print("  Note: noise scores the maximum entropy, but it carries no picture!\n")

# ---------- 6. Read and build the four versions ----------
img = cv2.imread(input_path)
if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()
img = resize_image(img, MAX_WIDTH)

wb       = white_balance(img, MAX_GAIN)
method_a = apply_clahe(wb, CLIP_LIMIT, TILE_GRID)
method_b = apply_clahe(auto_gamma_correction(wb, TARGET_MEAN), CLIP_LIMIT, TILE_GRID)
method_c = unsharp_mask(method_b, SIGMA, AMOUNT)

versions = [("Original", img),
            ("Method A", method_a),
            ("Method B", method_b),
            ("Method C", method_c)]

# ---------- 7. Calculate the metrics and print the table ----------
rows = []
print("| Method   | Brightness | Contrast | Entropy |")
print("|----------|-----------:|---------:|--------:|")
for name, image in versions:
    br, co, en = basic_metrics(image)
    rows.append((name, br, co, en))
    print(f"| {name:<8} | {br:>10.2f} | {co:>8.2f} | {en:>7.3f} |")

# ---------- 8. Save the table as a CSV file (opens in Excel) ----------
csv_path = os.path.join(results_folder, "step11a_metrics_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Brightness", "Contrast", "Entropy"])
    for name, br, co, en in rows:
        writer.writerow([name, round(br, 2), round(co, 2), round(en, 3)])
print("\nTable saved to:", os.path.abspath(csv_path))

# ---------- 9. Bar charts ----------
names = [r[0] for r in rows]
fig = plt.figure(figsize=(14, 4))
for i, (title, col) in enumerate([("Mean brightness", 1),
                                  ("Contrast (std)", 2),
                                  ("Entropy (bits)", 3)]):
    plt.subplot(1, 3, i + 1)
    values = [r[col] for r in rows]
    bars = plt.bar(names, values, color=["gray", "tab:blue", "tab:green", "tab:orange"])
    plt.title(title)
    for bar, v in zip(bars, values):                  # write the value on each bar
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(results_folder, "step11a_metric_bars.png"), dpi=150)

# ---------- 10. Show the four images so you can compare numbers with your eyes ----------
fig2 = plt.figure(figsize=(14, 4))
for i, (name, image) in enumerate(versions):
    plt.subplot(1, 4, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name)
    plt.axis("off")
plt.tight_layout()

plt.show()
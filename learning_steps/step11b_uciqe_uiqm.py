# step11b_uciqe_uiqm.py
# Purpose: add UCIQE and UIQM (simplified versions) to our metrics,
# build the full comparison table, and demonstrate PSNR/SSIM limits.

import cv2
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------- 1. Paths ----------
base_folder    = os.path.dirname(os.path.abspath(__file__))
input_path     = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")
results_folder = os.path.join(base_folder, "..", "results")
os.makedirs(results_folder, exist_ok=True)

# ---------- 2. Settings ----------
MAX_WIDTH, MAX_GAIN, TARGET_MEAN = 800, 3.0, 110
CLIP_LIMIT, TILE_GRID, SIGMA, AMOUNT = 2.0, (8, 8), 1.5, 0.8

# ---------- 3. Enhancement functions (same as Step 11A) ----------
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
    return cv2.LUT(image, np.clip(np.round(table), 0, 255).astype(np.uint8))

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

# ---------- 4. Basic metrics (from Step 11A) ----------
def mean_brightness(image):
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())

def contrast_std(image):
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std())

def entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# ---------- 5. NEW: UCIQE ----------
def uciqe(image):
    """Simplified UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s"""
    # float image in 0-1 -> OpenCV gives L in 0..100, a and b about -127..127
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0] / 100.0                    # lightness scaled to 0-1
    a = lab[:, :, 1] / 100.0                    # color channels scaled the same way
    b = lab[:, :, 2] / 100.0

    chroma  = np.sqrt(a ** 2 + b ** 2)          # color strength of each pixel
    sigma_c = chroma.std()                      # spread of chroma

    l_sorted = np.sort(L.ravel())               # all lightness values, small -> big
    n = max(1, int(0.01 * l_sorted.size))       # number of pixels in 1%
    con_l = l_sorted[-n:].mean() - l_sorted[:n].mean()   # brightest 1% minus darkest 1%

    saturation = chroma / np.sqrt(chroma ** 2 + L ** 2 + 1e-6)
    mu_s = saturation.mean()

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)

# ---------- 6. NEW: UIQM (simplified) ----------
def _trimmed_mean_var(x, alpha=0.1):
    """Mean and variance after dropping the lowest and highest 10% of values."""
    x = np.sort(x.ravel())
    n = x.size
    trimmed = x[int(alpha * n): int(n - alpha * n)]
    return trimmed.mean(), trimmed.var()

def uicm(image):
    """Colorfulness measure."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(image)]
    rg = r - g                                   # red-green signal
    yb = (r + g) / 2.0 - b                       # yellow-blue signal
    mu_rg, var_rg = _trimmed_mean_var(rg)
    mu_yb, var_yb = _trimmed_mean_var(yb)
    return float(-0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2)
                 + 0.1586 * np.sqrt(var_rg + var_yb))

def _block_view(ch, block=8):
    """Cut a 2D array into 8x8 blocks -> shape (rows, block, cols, block)."""
    h, w = ch.shape
    kh, kw = h // block, w // block
    return ch[:kh * block, :kw * block].reshape(kh, block, kw, block), kh, kw

def _eme(ch, block=8):
    """Enhancement measure estimation: average log(max/min) over blocks."""
    blocks, kh, kw = _block_view(ch, block)
    bmax = blocks.max(axis=(1, 3))
    bmin = blocks.min(axis=(1, 3))
    valid = (bmax > 0) & (bmin > 0)              # avoid division by zero / log(0)
    return 2.0 / (kh * kw) * np.sum(np.log(bmax[valid] / bmin[valid]))

def uism(image):
    """Sharpness measure: EME on Sobel-edge-weighted channels."""
    weights = [0.114, 0.587, 0.299]              # B, G, R (OpenCV order)
    total = 0.0
    for ch, w in zip(cv2.split(image), weights):
        ch = ch.astype(np.float32)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1)
        edge = np.sqrt(gx ** 2 + gy ** 2)
        edge = edge / (edge.max() + 1e-6)        # scale edges to 0-1
        total += w * _eme(ch * edge)             # keep only edge regions
    return float(total)

def uiconm(image):
    """Contrast measure (simplified: no PLIP arithmetic)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blocks, kh, kw = _block_view(gray, 8)
    bmax = blocks.max(axis=(1, 3))
    bmin = blocks.min(axis=(1, 3))
    top, bot = bmax - bmin, bmax + bmin
    valid = (top > 0) & (bot > 0)
    ratio = top[valid] / bot[valid]              # between 0 and 1
    return float(np.abs(np.sum(ratio * np.log(ratio))) / (kh * kw))

def uiqm(image):
    """Return (UIQM, UICM, UISM, UIConM). UIQM = 0.0282*UICM + 0.2953*UISM + 3.5753*UIConM"""
    c, s, k = uicm(image), uism(image), uiconm(image)
    return 0.0282 * c + 0.2953 * s + 3.5753 * k, c, s, k

def all_metrics(image):
    """Return (brightness, contrast, entropy, UCIQE, UIQM)."""
    return (mean_brightness(image), contrast_std(image), entropy(image),
            uciqe(image), uiqm(image)[0])

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 7. Read and build the versions ----------
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

versions = [("Original", img), ("Method A", method_a),
            ("Method B", method_b), ("Method C", method_c)]

# ---------- 8. Sanity test: does UIQM react to blur? ----------
blurred = cv2.GaussianBlur(img, (0, 0), 3)
_, c1, s1, k1 = uiqm(img)
_, c2, s2, k2 = uiqm(blurred)
print("SANITY TEST: original vs blurred copy")
print(f"  UISM    (sharpness): {s1:.3f} -> {s2:.3f}   (should drop)")
print(f"  UIConM  (contrast) : {k1:.3f} -> {k2:.3f}   (should drop)")
print(f"  UCIQE              : {uciqe(img):.4f} -> {uciqe(blurred):.4f}   (may barely change)\n")

# ---------- 9. Full comparison table ----------
rows = []
print("| Method   | Brightness | Contrast | Entropy |  UCIQE |   UIQM |")
print("|----------|-----------:|---------:|--------:|-------:|-------:|")
for name, image in versions:
    br, co, en, uq, ui = all_metrics(image)
    rows.append((name, br, co, en, uq, ui))
    print(f"| {name:<8} | {br:>10.2f} | {co:>8.2f} | {en:>7.3f} | {uq:>6.4f} | {ui:>6.3f} |")

print("\nUIQM components (UICM, UISM, UIConM):")
for name, image in versions:
    _, c, s, k = uiqm(image)
    print(f"  {name:<9} UICM = {c:8.3f}   UISM = {s:6.3f}   UIConM = {k:6.3f}")

csv_path = os.path.join(results_folder, "step11b_full_metrics_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Brightness", "Contrast", "Entropy", "UCIQE", "UIQM"])
    for name, br, co, en, uq, ui in rows:
        writer.writerow([name, round(br, 2), round(co, 2), round(en, 3), round(uq, 4), round(ui, 3)])
print("\nTable saved to:", os.path.abspath(csv_path))

# ---------- 10. PSNR / SSIM demonstration ----------
print("\nPSNR / SSIM of each version compared with the ORIGINAL underwater image:")
print("(There is no true reference here, so these numbers only measure 'how different'.)")
for name, image in versions[1:]:
    p = peak_signal_noise_ratio(img, image, data_range=255)
    s = structural_similarity(img, image, channel_axis=2, data_range=255)
    print(f"  {name}: PSNR = {p:6.2f} dB   SSIM = {s:.3f}")

# ---------- 11. Bar charts for all five metrics ----------
titles = ["Brightness", "Contrast", "Entropy", "UCIQE", "UIQM"]
names  = [r[0] for r in rows]
fig = plt.figure(figsize=(18, 4))
for i, title in enumerate(titles):
    plt.subplot(1, 5, i + 1)
    values = [r[i + 1] for r in rows]
    bars = plt.bar(names, values, color=["gray", "tab:blue", "tab:green", "tab:orange"])
    plt.title(title)
    plt.xticks(rotation=30, fontsize=8)
    plt.ylim(0, max(values) * 1.15)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(results_folder, "step11b_metric_bars.png"), dpi=150)

fig2 = plt.figure(figsize=(14, 4))
for i, (name, image) in enumerate(versions):
    plt.subplot(1, 4, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name)
    plt.axis("off")
plt.tight_layout()

plt.show()
# dip_functions.py
# Shared functions: enhancement techniques (Steps 5-9) and quality metrics (Steps 11A-11B).
# This file only DEFINES functions. Running it directly does nothing.

import cv2
import numpy as np

# ===================== ENHANCEMENT =====================
def resize_image(image, max_width=800):
    """Shrink image so width <= max_width. Keeps aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    return cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)

def white_balance(image, max_gain=3.0):
    """Gray World white balance with a gain limit."""
    img_f = image.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    gains = np.minimum(means.mean() / means, max_gain)
    return np.clip(img_f * gains, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma):
    """Gamma correction with a lookup table. gamma < 1 brightens."""
    table = np.array([255 * (i / 255.0) ** gamma for i in range(256)])
    return cv2.LUT(image, np.clip(np.round(table), 0, 255).astype(np.uint8))

def auto_gamma_correction(image, target_mean=110, low=0.4, high=1.0):
    """Choose gamma from the image brightness, then apply it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.clip(gray.mean(), 1, 254)
    gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(mean / 255.0), low, high))
    return gamma_correction(image, gamma)

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

# ===================== BASIC METRICS =====================
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

# ===================== UCIQE =====================
def uciqe(image):
    """Simplified UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s"""
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0] / 100.0
    a = lab[:, :, 1] / 100.0
    b = lab[:, :, 2] / 100.0
    chroma  = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()
    l_sorted = np.sort(L.ravel())
    n = max(1, int(0.01 * l_sorted.size))
    con_l = l_sorted[-n:].mean() - l_sorted[:n].mean()
    saturation = chroma / np.sqrt(chroma ** 2 + L ** 2 + 1e-6)
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * saturation.mean())

# ===================== UIQM (simplified) =====================
def _trimmed_mean_var(x, alpha=0.1):
    x = np.sort(x.ravel())
    n = x.size
    trimmed = x[int(alpha * n): int(n - alpha * n)]
    return trimmed.mean(), trimmed.var()

def uicm(image):
    b, g, r = [c.astype(np.float32) for c in cv2.split(image)]
    rg = r - g
    yb = (r + g) / 2.0 - b
    mu_rg, var_rg = _trimmed_mean_var(rg)
    mu_yb, var_yb = _trimmed_mean_var(yb)
    return float(-0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2)
                 + 0.1586 * np.sqrt(var_rg + var_yb))

def _block_view(ch, block=8):
    h, w = ch.shape
    kh, kw = h // block, w // block
    return ch[:kh * block, :kw * block].reshape(kh, block, kw, block), kh, kw

def _eme(ch, block=8):
    blocks, kh, kw = _block_view(ch, block)
    bmax = blocks.max(axis=(1, 3))
    bmin = blocks.min(axis=(1, 3))
    valid = (bmax > 0) & (bmin > 0)
    return 2.0 / (kh * kw) * np.sum(np.log(bmax[valid] / bmin[valid]))

def uism(image):
    weights = [0.114, 0.587, 0.299]          # B, G, R
    total = 0.0
    for ch, w in zip(cv2.split(image), weights):
        ch = ch.astype(np.float32)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1)
        edge = np.sqrt(gx ** 2 + gy ** 2)
        edge = edge / (edge.max() + 1e-6)
        total += w * _eme(ch * edge)
    return float(total)

def uiconm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blocks, kh, kw = _block_view(gray, 8)
    bmax = blocks.max(axis=(1, 3))
    bmin = blocks.min(axis=(1, 3))
    top, bot = bmax - bmin, bmax + bmin
    valid = (top > 0) & (bot > 0)
    ratio = top[valid] / bot[valid]
    return float(np.abs(np.sum(ratio * np.log(ratio))) / (kh * kw))

def uiqm(image):
    """Return (UIQM, UICM, UISM, UIConM)."""
    c, s, k = uicm(image), uism(image), uiconm(image)
    return 0.0282 * c + 0.2953 * s + 3.5753 * k, c, s, k

def all_metrics(image):
    """Return (brightness, contrast, entropy, UCIQE, UIQM)."""
    return (mean_brightness(image), contrast_std(image), entropy(image),
            uciqe(image), uiqm(image)[0])
# metrics.py
# Image quality metrics (Steps 11A and 11B).
# UCIQE and UIQM are simplified versions: compare only inside this project.

import cv2
import numpy as np


# ---------- Basic metrics ----------
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


# ---------- UCIQE ----------
def uciqe(image):
    """Simplified UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s"""
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0] / 100.0
    a = lab[:, :, 1] / 100.0
    b = lab[:, :, 2] / 100.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()
    l_sorted = np.sort(L.ravel())
    n = max(1, int(0.01 * l_sorted.size))
    con_l = l_sorted[-n:].mean() - l_sorted[:n].mean()
    saturation = chroma / np.sqrt(chroma ** 2 + L ** 2 + 1e-6)
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * saturation.mean())


# ---------- UIQM (simplified) ----------
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


# ---------- Combined ----------
def calculate_metrics(image):
    """Return a dictionary: Brightness, Contrast, Entropy, UCIQE, UIQM."""
    return {"Brightness": mean_brightness(image),
            "Contrast":   contrast_std(image),
            "Entropy":    entropy(image),
            "UCIQE":      uciqe(image),
            "UIQM":       uiqm(image)[0]}
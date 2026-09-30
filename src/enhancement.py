# enhancement.py
# Brightness, contrast and detail enhancement (Steps 7, 8, 9).

import cv2
import numpy as np


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
    """CLAHE on the L channel of LAB, so colors are not distorted."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def unsharp_mask(image, sigma=1.5, amount=1.0):
    """Unsharp masking: sharpened = (1 + amount) * image - amount * blurred."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def sharpen_image(image, sigma=1.5, amount=0.8):
    """Clear-named wrapper around unsharp_mask (name from our project plan)."""
    return unsharp_mask(image, sigma, amount)
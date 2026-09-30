# preprocessing.py
# Cleanup done before enhancement: resizing and optional denoising.

import cv2


def resize_image(image, max_width=800):
    """Shrink image so its width is at most max_width. Keeps the aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    return cv2.resize(image, (max_width, int(h * max_width / w)),
                      interpolation=cv2.INTER_AREA)


def denoise_image(image, method="median", ksize=3):
    """
    Optional noise reduction (Step 5).
    method : "median" for dot-like noise, "gaussian" for grainy noise
    ksize  : odd kernel size (3, 5, 7...)
    """
    if method == "median":
        return cv2.medianBlur(image, ksize)
    return cv2.GaussianBlur(image, (ksize, ksize), 0)
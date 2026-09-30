# pipeline.py
# The enhancement pipeline: load an image, enhance it, measure it, save it.
# This file only DEFINES functions. Run main.py to use them.

import os
import cv2

from dip_functions import (resize_image, white_balance, auto_gamma_correction,
                           apply_clahe, unsharp_mask, all_metrics)

# ---------- Default settings (all parameters in one place) ----------
DEFAULT_PARAMS = {
    "max_width":   800,       # resize width
    "max_gain":    3.0,       # white balance gain limit
    "target_mean": 110,       # brightness target for automatic gamma
    "clip_limit":  2.0,       # CLAHE strength
    "tile_grid":   (8, 8),    # CLAHE tiles
    "sigma":       1.5,       # sharpening blur size
    "amount":      0.8,       # sharpening strength
}


def load_image(path, max_width=800):
    """
    Read an image from disk and resize it if it is too wide.

    path      : full path of the image file (text)
    max_width : maximum width in pixels (number)
    Returns   : the image as a BGR NumPy array
    Raises    : FileNotFoundError if the image cannot be read
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {os.path.abspath(path)}")
    return resize_image(image, max_width)


def sharpen_image(image, sigma=1.5, amount=0.8):
    """
    Sharpen an image (unsharp masking).

    image  : BGR image
    sigma  : blur size used to find details
    amount : sharpening strength
    Returns: the sharpened image
    """
    return unsharp_mask(image, sigma, amount)


def calculate_metrics(image):
    """
    Measure the quality of one image.

    image   : BGR image
    Returns : a dictionary with Brightness, Contrast, Entropy, UCIQE, UIQM
    """
    br, co, en, uq, ui = all_metrics(image)
    return {"Brightness": br, "Contrast": co, "Entropy": en,
            "UCIQE": uq, "UIQM": ui}


def enhance_image(image, params=None):
    """
    Run the full enhancement pipeline:
    White Balance -> Gamma -> CLAHE -> Sharpen

    image   : BGR image (already loaded and resized)
    params  : optional dictionary that overrides DEFAULT_PARAMS,
              for example {"clip_limit": 3.0}
    Returns : (final_image, stages)
              stages is a dictionary holding every intermediate image
    """
    p = dict(DEFAULT_PARAMS)        # make a copy of the defaults
    if params is not None:
        p.update(params)            # replace only the values the caller gave

    stages = {"Original": image}
    stages["Color Corrected"]   = white_balance(image, p["max_gain"])
    stages["Gamma Corrected"]   = auto_gamma_correction(stages["Color Corrected"],
                                                        p["target_mean"])
    stages["Contrast Enhanced"] = apply_clahe(stages["Gamma Corrected"],
                                              p["clip_limit"], p["tile_grid"])
    stages["Final Enhanced"]    = sharpen_image(stages["Contrast Enhanced"],
                                                p["sigma"], p["amount"])
    return stages["Final Enhanced"], stages


def save_image(image, path):
    """Save an image, creating the folder first if it does not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
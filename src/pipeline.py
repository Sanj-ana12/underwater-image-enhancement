# pipeline.py
# The fixed enhancement pipeline: load, enhance, save.

import os
import cv2

from src.preprocessing import resize_image
from src.color_correction import white_balance
from src.enhancement import auto_gamma_correction, apply_clahe, sharpen_image

DEFAULT_PARAMS = {
    "max_width":   800,
    "max_gain":    3.0,
    "target_mean": 110,
    "clip_limit":  2.0,
    "tile_grid":   (8, 8),
    "sigma":       1.5,
    "amount":      0.8,
}


def load_image(path, max_width=800):
    """Read an image and resize it. Raises FileNotFoundError if it cannot be read."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {os.path.abspath(path)}")
    return resize_image(image, max_width)


def enhance_image(image, params=None):
    """
    Fixed pipeline: White Balance -> Gamma -> CLAHE -> Sharpen.
    Returns (final_image, stages_dictionary).
    """
    p = dict(DEFAULT_PARAMS)
    if params is not None:
        p.update(params)

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
    """Save an image, creating the folder first if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
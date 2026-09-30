# color_correction.py
# Fixes the blue/green color cast (Step 6).

import numpy as np


def white_balance(image, max_gain=3.0):
    """Gray World white balance with a gain limit. Returns the corrected image."""
    img_f = image.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)               # B, G, R means
    gains = np.minimum(means.mean() / means, max_gain)      # capped gains
    return np.clip(img_f * gains, 0, 255).astype(np.uint8)
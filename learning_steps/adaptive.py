# adaptive.py
# Adaptive enhancement: measure the image, then choose which techniques
# to apply and how strongly. Each decision is written into a list so we
# can print it (and show it in the Streamlit app later).

import cv2
import numpy as np

from dip_functions import (white_balance, gamma_correction,
                           apply_clahe, unsharp_mask)

# ---------- All decision thresholds in one place (tune these!) ----------
T = {
    "cast_none":     0.10,   # imbalance below this  -> no color cast
    "cast_mild":     0.25,   # below this            -> mild cast
    "cast_moderate": 0.45,   # below this            -> moderate cast, else strong
    "dark":          70,     # brightness below this -> very dark
    "bright_ok":     115,    # brightness at/above   -> bright enough, skip gamma
    "contrast_low":  35,     # contrast below this   -> low contrast
    "contrast_ok":   55,
    "contrast_high": 65,     # contrast at/above     -> skip CLAHE
    "sharp_high":    300,    # sharpness at/above    -> already sharp/noisy
}


def analyze_image(image):
    """
    Measure simple statistics of a BGR image.
    Returns a dictionary: brightness, contrast, imbalance, rb_ratio, sharpness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    means = image.reshape(-1, 3).astype(np.float32).mean(axis=0)   # B, G, R means
    imbalance = (means.max() - means.min()) / (means.mean() + 1e-6)
    return {
        "brightness": float(gray.mean()),
        "contrast":   float(gray.std()),
        "imbalance":  float(imbalance),
        "rb_ratio":   float(means[2] / (means[0] + 1e-6)),
        "sharpness":  float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def enhance_adaptive(image):
    """
    Adaptive pipeline: White Balance -> Gamma -> CLAHE -> Sharpen,
    where every step is chosen from the CURRENT image statistics.

    image   : BGR image (already loaded and resized)
    Returns : (final_image, stages, decisions)
              stages    : dictionary of intermediate images (same names as enhance_image)
              decisions : list of text lines explaining each choice
    """
    stages = {"Original": image}
    decisions = []
    current = image

    # ---- 1. Color correction ----
    s = analyze_image(current)
    imb = s["imbalance"]
    if imb < T["cast_none"]:
        decisions.append(f"Color: imbalance {imb:.2f} (R/B {s['rb_ratio']:.2f}) -> no cast, white balance SKIPPED")
    else:
        if imb < T["cast_mild"]:
            level, gain = "mild", 1.5
        elif imb < T["cast_moderate"]:
            level, gain = "moderate", 2.0
        else:
            level, gain = "strong", 3.0
        current = white_balance(current, gain)
        decisions.append(f"Color: imbalance {imb:.2f} (R/B {s['rb_ratio']:.2f}) -> {level} cast, white balance with max_gain {gain}")
    stages["Color Corrected"] = current

    # ---- 2. Brightness (measured AFTER color correction) ----
    s = analyze_image(current)
    b = s["brightness"]
    if b >= T["bright_ok"]:
        decisions.append(f"Brightness: {b:.0f} -> bright enough, gamma SKIPPED")
    else:
        target = 120 if b < T["dark"] else 110
        b_safe = min(max(b, 1.0), 254.0)                       # avoid log(0)
        gamma = float(np.clip(np.log(target / 255.0) / np.log(b_safe / 255.0), 0.4, 1.0))
        current = gamma_correction(current, gamma)
        decisions.append(f"Brightness: {b:.0f} -> {'very dark' if b < T['dark'] else 'dark'}, gamma {gamma:.2f} (target {target})")
    stages["Gamma Corrected"] = current

    # ---- 3. Contrast (measured AFTER gamma) ----
    s = analyze_image(current)
    c = s["contrast"]
    if c >= T["contrast_high"]:
        decisions.append(f"Contrast: {c:.0f} -> already high, CLAHE SKIPPED")
    else:
        if c < T["contrast_low"]:
            clip = 3.0
        elif c < T["contrast_ok"]:
            clip = 2.0
        else:
            clip = 1.5
        current = apply_clahe(current, clip, (8, 8))
        decisions.append(f"Contrast: {c:.0f} -> CLAHE with clip_limit {clip}")
    stages["Contrast Enhanced"] = current

    # ---- 4. Sharpening (measured AFTER CLAHE) ----
    s = analyze_image(current)
    sh = s["sharpness"]
    amount = 0.4 if sh >= T["sharp_high"] else 0.8
    current = unsharp_mask(current, 1.5, amount)
    decisions.append(f"Sharpness: {sh:.0f} -> unsharp mask amount {amount}")
    stages["Final Enhanced"] = current

    return current, stages, decisions
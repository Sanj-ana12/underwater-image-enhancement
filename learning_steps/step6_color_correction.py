# step6_color_correction.py
# Purpose: fix the blue/green color cast using the Gray World assumption.
# We compare: Original, Basic Gray World, Gray World with gain limit.
# We also show RGB channel statistics before and after.

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Paths ----------
base_folder    = os.path.dirname(os.path.abspath(__file__))
input_path     = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")
output_folder  = os.path.join(base_folder, "..", "output_images")
results_folder = os.path.join(base_folder, "..", "results")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# ---------- 2. Settings you can change ----------
MAX_WIDTH = 800     # resize big images to this width
MAX_GAIN  = 3.0     # largest multiplier allowed in the limited version

# ---------- 3. Helper functions ----------
def resize_image(image, max_width):
    """Shrink image so width <= max_width. Keeps aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def gray_world(image, max_gain=None):
    """
    White balance using the Gray World assumption.
    image    : BGR image (uint8)
    max_gain : if given, no channel gain can be bigger than this
    Returns  : (corrected image, gains array in B, G, R order)
    """
    img_f = image.astype(np.float32)               # decimals for the math

    # mean of each channel -> array of 3 numbers [B, G, R]
    means = img_f.reshape(-1, 3).mean(axis=0)

    gray_mean = means.mean()                       # target level
    gains = gray_mean / means                      # one gain per channel

    if max_gain is not None:
        gains = np.minimum(gains, max_gain)        # cap the gains

    corrected = img_f * gains                      # multiply each channel by its gain
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected, gains

def channel_stats(image):
    """Return (means, stds) for B, G, R channels."""
    flat = image.reshape(-1, 3).astype(np.float32)
    return flat.mean(axis=0), flat.std(axis=0)

def to_rgb(image):
    """BGR -> RGB for Matplotlib."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 4. Read and resize ----------
img = cv2.imread(input_path)
if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()

img = resize_image(img, MAX_WIDTH)

# ---------- 5. Apply the two versions ----------
gw_basic,   gains_basic   = gray_world(img)
gw_limited, gains_limited = gray_world(img, max_gain=MAX_GAIN)

print("Gains used (Blue, Green, Red)")
print("  Basic Gray World   :", np.round(gains_basic, 2))
print("  Limited Gray World :", np.round(gains_limited, 2))

# ---------- 6. Print channel statistics ----------
versions = [("Original", img),
            ("Gray World", gw_basic),
            (f"Gray World (max gain {MAX_GAIN})", gw_limited)]

print("\nChannel statistics (mean / std)")
print(f"{'Version':<28}{'Blue':>16}{'Green':>16}{'Red':>16}")
all_means = []
for name, image in versions:
    m, s = channel_stats(image)
    all_means.append(m)
    print(f"{name:<28}"
          f"{m[0]:>8.1f}/{s[0]:<6.1f}"
          f"{m[1]:>8.1f}/{s[1]:<6.1f}"
          f"{m[2]:>8.1f}/{s[2]:<6.1f}")

# ---------- 7. Save corrected images ----------
cv2.imwrite(os.path.join(output_folder, "step6_gray_world.jpg"),         gw_basic)
cv2.imwrite(os.path.join(output_folder, "step6_gray_world_limited.jpg"), gw_limited)

# ---------- 8. Figure 1: images (top) and histograms (bottom) ----------
fig1 = plt.figure(figsize=(14, 7))
for i, (name, image) in enumerate(versions):
    plt.subplot(2, 3, i + 1)
    plt.imshow(to_rgb(image))
    plt.title(name)
    plt.axis("off")

    plt.subplot(2, 3, i + 4)
    for ch_index, color in enumerate(["blue", "green", "red"]):
        hist = cv2.calcHist([image], [ch_index], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 255])
    plt.xlabel("Pixel value")
    plt.ylabel("Pixels")
plt.tight_layout()
fig1.savefig(os.path.join(results_folder, "step6_color_correction.png"), dpi=150)

# ---------- 9. Figure 2: bar chart of channel means ----------
fig2 = plt.figure(figsize=(8, 5))
x = np.arange(3)                                   # positions for B, G, R
width = 0.25
for i, ((name, _), means) in enumerate(zip(versions, all_means)):
    plt.bar(x + i * width, means, width, label=name)
plt.xticks(x + width, ["Blue", "Green", "Red"])
plt.ylabel("Mean pixel value")
plt.title("Channel means before and after color correction")
plt.legend()
plt.tight_layout()
fig2.savefig(os.path.join(results_folder, "step6_channel_means.png"), dpi=150)

plt.show()

# step3_analyze_image.py
# Purpose: analyze an underwater image BEFORE enhancement.
# We calculate channel statistics, brightness, contrast,
# and draw histograms.

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Build file paths ----------
base_folder    = os.path.dirname(os.path.abspath(__file__))
input_path     = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")
results_folder = os.path.join(base_folder, "..", "results")

# Create the results folder if it does not exist
os.makedirs(results_folder, exist_ok=True)

# ---------- 2. Read the image ----------
img = cv2.imread(input_path)

if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()

# ---------- 3. Image dimensions ----------
height, width, channels = img.shape
print("========== IMAGE DIMENSIONS ==========")
print("Width :", width, "| Height :", height, "| Channels :", channels)

# ---------- 4. Split into B, G, R channels ----------
# OpenCV order is Blue, Green, Red
b, g, r = cv2.split(img)

channel_list  = [b, g, r]
channel_names = ["Blue", "Green", "Red"]
plot_colors   = ["blue", "green", "red"]

# ---------- 5. Channel statistics ----------
print("\n========== CHANNEL STATISTICS ==========")
means = []                                   # will store the mean of each channel

for name, ch in zip(channel_names, channel_list):
    ch_mean = np.mean(ch)
    ch_std  = np.std(ch)
    ch_min  = np.min(ch)
    ch_max  = np.max(ch)
    means.append(ch_mean)
    print(f"{name:<6} mean = {ch_mean:6.2f} | std = {ch_std:6.2f} | min = {ch_min:3d} | max = {ch_max:3d}")

# ---------- 6. Brightness and contrast (from grayscale) ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mean_brightness = np.mean(gray)
contrast        = np.std(gray)

print("\n========== BRIGHTNESS AND CONTRAST ==========")
print(f"Mean brightness (0-255) : {mean_brightness:.2f}")
print(f"Contrast (std of gray)  : {contrast:.2f}")

if mean_brightness < 80:
    print("Brightness verdict : DARK image")
elif mean_brightness < 160:
    print("Brightness verdict : MEDIUM brightness")
else:
    print("Brightness verdict : BRIGHT image")

if contrast < 40:
    print("Contrast verdict   : LOW contrast")
else:
    print("Contrast verdict   : ACCEPTABLE contrast")

# ---------- 7. Color cast analysis ----------
print("\n========== COLOR CAST ANALYSIS ==========")
dominant_index = int(np.argmax(means))       # index of the largest mean
weakest_index  = int(np.argmin(means))       # index of the smallest mean
print("Dominant channel :", channel_names[dominant_index])
print("Weakest channel  :", channel_names[weakest_index])
print(f"Red mean / Blue mean  ratio : {means[2] / means[0]:.2f}")
print(f"Red mean / Green mean ratio : {means[2] / means[1]:.2f}")
print("(A ratio far below 1.0 means red has been lost.)")

# ---------- 8. Plot everything in one figure ----------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # for Matplotlib

plt.figure(figsize=(12, 8))

# (a) Original image
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Underwater Image")
plt.axis("off")

# (b) Histogram of each color channel
plt.subplot(2, 2, 2)
for name, ch, color in zip(channel_names, channel_list, plot_colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color=color, label=name)
plt.title("RGB Histograms")
plt.xlabel("Pixel value (0 = dark, 255 = bright)")
plt.ylabel("Number of pixels")
plt.xlim([0, 255])
plt.legend()

# (c) Histogram of the grayscale image
plt.subplot(2, 2, 3)
gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(gray_hist, color="black")
plt.title("Grayscale Histogram (Brightness)")
plt.xlabel("Pixel value (0 = dark, 255 = bright)")
plt.ylabel("Number of pixels")
plt.xlim([0, 255])

# (d) Bar chart of channel means (color distribution)
plt.subplot(2, 2, 4)
plt.bar(channel_names, means, color=plot_colors)
plt.title("Average Value of Each Channel")
plt.ylabel("Mean pixel value")
plt.ylim([0, 255])

plt.tight_layout()

# Save the figure into the results folder, then show it
save_path = os.path.join(results_folder, "step3_analysis_original.png")
plt.savefig(save_path, dpi=150)
print("\nAnalysis figure saved to:", os.path.abspath(save_path))

plt.show()
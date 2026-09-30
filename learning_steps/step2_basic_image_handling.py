# step2_basic_image_handling.py
# Purpose: read an image, get its properties, save it, and show it
#          with both OpenCV and Matplotlib.

import cv2                        # OpenCV: read/save/show images
import os                         # build file paths
import sys                        # lets us stop the program if needed
import matplotlib.pyplot as plt   # Matplotlib: plotting and image display

# ---------- 1. Build file paths ----------
base_folder = os.path.dirname(os.path.abspath(__file__))
input_path  = os.path.join(base_folder, "..", "input_images",  "underwater1.jpg")
output_path = os.path.join(base_folder, "..", "output_images", "saved_copy.jpg")

# ---------- 2. Read the image ----------
img = cv2.imread(input_path)

if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(input_path))
    sys.exit()          # stop the program here

# ---------- 3. Find image properties ----------
height, width, channels = img.shape   # shape is (rows, columns, channels)

print("----- Image Information -----")
print("Width         :", width, "pixels")
print("Height        :", height, "pixels")
print("Channels      :", channels)
print("Total pixels  :", width * height)
print("Data type     :", img.dtype)
print("Pixel at top-left corner (B, G, R):", img[100, 200])

# ---------- 4. Display with OpenCV ----------
cv2.imshow("Original (OpenCV window)", img)
cv2.waitKey(0)              # press any key to continue
cv2.destroyAllWindows()

# ---------- 5. Save the image ----------
cv2.imwrite(output_path, img)
print("Image saved to:", os.path.abspath(output_path))

# ---------- 6. Convert BGR to RGB for Matplotlib ----------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- 7. Display with Matplotlib (wrong vs correct) ----------
plt.figure(figsize=(10, 5))               # size of the whole figure

plt.subplot(1, 2, 1)                      # 1 row, 2 columns, position 1
plt.imshow(img)                           # BGR shown as if RGB -> wrong colors
plt.title("Wrong: BGR shown directly")
plt.axis("off")                           # hide the axis numbers

plt.subplot(1, 2, 2)                      # position 2
plt.imshow(img_rgb)                       # correct colors
plt.title("Correct: converted to RGB")
plt.axis("off")

plt.tight_layout()                        # avoid overlapping text
plt.show()                                # open the Matplotlib window
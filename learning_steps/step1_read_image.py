# step1_read_image.py
# Purpose: read an underwater image from disk and display it on screen.

import cv2      # OpenCV: for reading and showing images
import os       # os: for building file paths safely

# --- 1. Build the path to the image ---
# __file__ is the location of this Python file (inside the "code" folder).
# We go one folder up (..) and then into "input_images".
base_folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_folder, "..", "input_images", "underwater1.jpg")

# --- 2. Read the image ---
img = cv2.imread(image_path)

# --- 3. Check whether the image was really loaded ---
if img is None:
    print("ERROR: Could not read the image.")
    print("Path tried:", os.path.abspath(image_path))
    print("Check that the file exists and the name is correct.")
else:
    print("Image loaded successfully!")

    # --- 4. Show the image in a window ---
    cv2.imshow("Original Underwater Image", img)

    # --- 5. Wait until you press any key, then close the window ---
    cv2.waitKey(0)
    cv2.destroyAllWindows()
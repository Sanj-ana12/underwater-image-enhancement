# verify_setup.py
# Purpose: check that all required libraries are installed correctly.

import cv2                 # OpenCV
import numpy as np         # NumPy
import matplotlib          # Matplotlib
import skimage             # scikit-image

print("OpenCV version       :", cv2.__version__)
print("NumPy version        :", np.__version__)
print("Matplotlib version   :", matplotlib.__version__)
print("scikit-image version :", skimage.__version__)
print("Everything is installed correctly!")
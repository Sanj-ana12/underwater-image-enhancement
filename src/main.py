# main.py
# Batch program: enhance every image in data/input/ and report the results.
# Run with:  python src/main.py     (from the project root folder)

import os
import sys
import csv

# Project root = the folder ABOVE src/. Adding it to Python's search path
# lets "from src.xxx import ..." work no matter how the file is started.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import load_image, enhance_image, save_image
from src.adaptive import enhance_adaptive
from src.metrics import calculate_metrics

USE_ADAPTIVE = True                                  # False = fixed pipeline
IMAGE_TYPES  = (".jpg", ".jpeg", ".png", ".bmp")
METRIC_NAMES = ["Brightness", "Contrast", "Entropy", "UCIQE", "UIQM"]


def main():
    input_folder    = os.path.join(ROOT, "data", "input")
    enhanced_folder = os.path.join(ROOT, "outputs", "enhanced")
    results_folder  = os.path.join(ROOT, "outputs", "results")
    os.makedirs(results_folder, exist_ok=True)

    if not os.path.isdir(input_folder):
        print("Folder not found:", input_folder)
        return

    files = sorted(f for f in os.listdir(input_folder)
                   if f.lower().endswith(IMAGE_TYPES))
    if not files:
        print("No images found in:", input_folder)
        return

    mode = "adaptive" if USE_ADAPTIVE else "fixed"
    print(f"Found {len(files)} image(s). Mode: {mode}\n")

    table_rows = []
    for filename in files:
        name = os.path.splitext(filename)[0]
        try:
            image = load_image(os.path.join(input_folder, filename))
        except FileNotFoundError as error:
            print("Skipped:", error)
            continue

        if USE_ADAPTIVE:
            final, stages, decisions = enhance_adaptive(image)
        else:
            final, stages = enhance_image(image)
            decisions = []

        save_image(final, os.path.join(enhanced_folder, f"{name}_enhanced.jpg"))

        before = calculate_metrics(image)
        after  = calculate_metrics(final)

        print(f"----- {filename} -----")
        for line in decisions:
            print("  -", line)
        print(f"{'':<10}" + "".join(f"{m:>12}" for m in METRIC_NAMES))
        print(f"{'Original':<10}" + "".join(f"{before[m]:>12.3f}" for m in METRIC_NAMES))
        print(f"{'Enhanced':<10}" + "".join(f"{after[m]:>12.3f}" for m in METRIC_NAMES))
        print()

        table_rows.append([filename, "Original"] + [round(before[m], 4) for m in METRIC_NAMES])
        table_rows.append([filename, "Enhanced"] + [round(after[m], 4) for m in METRIC_NAMES])

    csv_path = os.path.join(results_folder, f"batch_results_{mode}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Version"] + METRIC_NAMES)
        writer.writerows(table_rows)

    print("Enhanced images saved in:", enhanced_folder)
    print("Results table saved to  :", csv_path)


if __name__ == "__main__":
    main()
# main.py
# The main program: enhance every image in input_images/ and report the results.
# Run it with:  python code/main.py

import os
import csv

from pipeline import load_image, enhance_image, calculate_metrics, save_image

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")     # file types we accept


def main():
    base_folder   = os.path.dirname(os.path.abspath(__file__))
    input_folder  = os.path.join(base_folder, "..", "input_images")
    output_folder = os.path.join(base_folder, "..", "output_images")
    results_folder = os.path.join(base_folder, "..", "results")
    os.makedirs(results_folder, exist_ok=True)

    # Find all image files in the input folder
    files = sorted(f for f in os.listdir(input_folder)
                   if f.lower().endswith(IMAGE_TYPES))
    if not files:
        print("No images found in:", os.path.abspath(input_folder))
        return                                       # stop the function

    print(f"Found {len(files)} image(s). Enhancing...\n")

    metric_names = ["Brightness", "Contrast", "Entropy", "UCIQE", "UIQM"]
    table_rows = []

    for filename in files:
        name = os.path.splitext(filename)[0]         # "underwater1.jpg" -> "underwater1"

        try:
            image = load_image(os.path.join(input_folder, filename))
        except FileNotFoundError as error:
            print("Skipped:", error)
            continue                                 # go to the next file

        final, stages = enhance_image(image)
        save_image(final, os.path.join(output_folder, f"{name}_enhanced.jpg"))

        before = calculate_metrics(image)
        after  = calculate_metrics(final)

        print(f"----- {filename} -----")
        print(f"{'':<10}" + "".join(f"{m:>12}" for m in metric_names))
        print(f"{'Original':<10}" + "".join(f"{before[m]:>12.3f}" for m in metric_names))
        print(f"{'Enhanced':<10}" + "".join(f"{after[m]:>12.3f}" for m in metric_names))
        print()

        table_rows.append([filename, "Original"] + [round(before[m], 4) for m in metric_names])
        table_rows.append([filename, "Enhanced"] + [round(after[m], 4) for m in metric_names])

    # Save all results in one CSV file
    csv_path = os.path.join(results_folder, "step13_batch_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Version"] + metric_names)
        writer.writerows(table_rows)

    print("Enhanced images saved in:", os.path.abspath(output_folder))
    print("Results table saved to  :", os.path.abspath(csv_path))


if __name__ == "__main__":
    main()
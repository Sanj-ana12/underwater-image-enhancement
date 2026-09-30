# Underwater Image Enhancement Using Digital Image Processing

Team: [names]   |   College: [name]   |   Guide: [name]

## What it does
Takes an underwater image, measures its problems (color cast, darkness,
low contrast), enhances it with classical image processing techniques
(white balance, gamma correction, CLAHE, unsharp masking), and evaluates the
result with no-reference quality metrics.

## Features
- Fixed pipeline and adaptive pipeline (parameters chosen from image statistics)
- Metrics: brightness, contrast, entropy, UCIQE, UIQM (simplified versions)
- Streamlit web interface with before/after comparison and download

## Setup
pip install -r requirements.txt

## Run
- Batch enhancement:  python src/main.py
- Web app:            streamlit run app.py

## Folders
- data/input   : images to enhance
- outputs      : enhanced images and result tables
- src          : source code

## Limitations
UCIQE/UIQM here are simplified, so values are comparable only within this
project. Rule-based thresholds were tuned on our own sample images.
# app.py
# Streamlit interface for underwater image enhancement.
# Run with:  streamlit run code/app.py

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import resize_image
from src.pipeline import enhance_image
from src.metrics import calculate_metrics
from src.adaptive import enhance_adaptive

METRIC_NAMES = ["Brightness", "Contrast", "Entropy", "UCIQE", "UIQM"]


# ---------- Helper functions ----------
def decode_upload(uploaded_file, max_width=800):
    """Turn an uploaded file (bytes) into a resized BGR image, or None if invalid."""
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return resize_image(image, max_width)


def to_rgb(image):
    """BGR -> RGB so the browser shows correct colors."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_png_bytes(image):
    """Encode an image as PNG bytes for the download button."""
    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


def metrics_table(before, after):
    """Build a table: Original, Enhanced and % change for each metric."""
    rows = []
    for m in METRIC_NAMES:
        change = (after[m] / before[m] - 1) * 100 if before[m] != 0 else 0.0
        rows.append({"Metric": m,
                     "Original": round(before[m], 3),
                     "Enhanced": round(after[m], 3),
                     "Change (%)": round(change, 1)})
    return pd.DataFrame(rows).set_index("Metric")


# ---------- Page setup ----------
st.set_page_config(page_title="Underwater Image Enhancer", layout="wide")
st.title("Underwater Image Enhancement")
st.caption("Digital image processing: white balance, gamma correction, CLAHE and sharpening.")

# ---------- Sidebar: settings ----------
st.sidebar.header("Settings")
mode = st.sidebar.radio(
    "Enhancement mode",
    ["Adaptive (automatic)", "Fixed (manual)"],
    help="Adaptive measures the image and chooses settings. Fixed uses the values below.")

params = {}
if mode.startswith("Fixed"):
    params["max_gain"]   = st.sidebar.slider("White balance max gain", 1.0, 5.0, 3.0, 0.5)
    params["clip_limit"] = st.sidebar.slider("CLAHE clip limit", 1.0, 6.0, 2.0, 0.5)
    params["amount"]     = st.sidebar.slider("Sharpening amount", 0.0, 2.0, 0.8, 0.1)

# ---------- Upload ----------
uploaded = st.file_uploader("Upload an underwater image",
                            type=["jpg", "jpeg", "png", "bmp"])
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()                                   # stop the script here

# Forget the old result when a different file is uploaded
file_id = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("file_id") != file_id:
    st.session_state["file_id"] = file_id
    st.session_state.pop("result", None)

image = decode_upload(uploaded)
if image is None:
    st.error("This file could not be read as an image. Try another file.")
    st.stop()

# ---------- Show the original, then the Enhance button ----------
if "result" not in st.session_state:
    st.subheader("Original image")
    st.image(to_rgb(image), use_container_width=True)

if st.button("Enhance Image", type="primary"):
    with st.spinner("Enhancing..."):
        if mode.startswith("Adaptive"):
            final, stages, decisions = enhance_adaptive(image)
        else:
            final, stages = enhance_image(image, params)
            decisions = []
        st.session_state["result"] = {
            "final": final, "stages": stages, "decisions": decisions,
            "before": calculate_metrics(image), "after": calculate_metrics(final),
            "mode": mode,
        }

# ---------- Show the result (kept alive by session_state) ----------
if "result" in st.session_state:
    r = st.session_state["result"]
    st.success(f"Done. Mode used: {r['mode']}")

    st.subheader("Before / After")
    col1, col2 = st.columns(2)
    col1.image(to_rgb(image), caption="Original", use_container_width=True)
    col2.image(to_rgb(r["final"]), caption="Enhanced", use_container_width=True)

    st.subheader("Image quality metrics")
    st.dataframe(metrics_table(r["before"], r["after"]), use_container_width=True)
    st.caption("UCIQE and UIQM are simplified versions. Compare only inside this project. "
               "Always check the images by eye as well.")

    if r["decisions"]:
        with st.expander("What did the adaptive system decide?"):
            for line in r["decisions"]:
                st.write("- " + line)

    with st.expander("See every processing stage"):
        cols = st.columns(len(r["stages"]))
        for col, (name, stage_img) in zip(cols, r["stages"].items()):
            col.image(to_rgb(stage_img), caption=name, use_container_width=True)

    st.download_button("Download enhanced image",
                       data=to_png_bytes(r["final"]),
                       file_name="enhanced_image.png",
                       mime="image/png")
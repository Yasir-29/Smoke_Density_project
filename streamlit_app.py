from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit as st
from PIL import Image

from inference import predict_smoke_from_bytes


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "model.joblib"


def _b64png_to_pil(b64: str) -> Image.Image:
    """Decode base64 PNG returned by `inference.py`."""
    raw = base64.b64decode(b64.encode("ascii"))
    return Image.open(io.BytesIO(raw))


def _label_style(label_name: str) -> dict:
    if label_name == "Low":
        return {"kind": "success", "prefix": "Smoke: Low"}
    if label_name == "Moderate":
        return {"kind": "warning", "prefix": "Smoke: Moderate"}
    if label_name == "High":
        return {"kind": "error", "prefix": "Smoke: High"}
    return {"kind": "info", "prefix": "Smoke: Unknown"}


st.set_page_config(page_title="Smoke Detection", layout="wide")
st.title("Smoke Detection (DCP + RandomForest)")

if not MODEL_PATH.exists():
    st.error(f"Model not found at `{MODEL_PATH}`. Train first with `py main.py train ...`.")
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Best results if image is similar to your dataset (hazy/smoke).",
    )

    st.divider()
    camera_file = st.camera_input("Or take a photo (camera)")

    predict_btn = st.button("Predict smoke", type="primary", disabled=(uploaded_file is None and camera_file is None))

with right:
    st.subheader("Result")
    result_box = st.empty()
    preview_box = st.empty()


def _run_prediction(image_bytes: bytes) -> dict:
    if not image_bytes:
        raise ValueError("Empty image.")
    return predict_smoke_from_bytes(MODEL_PATH, image_bytes)


if predict_btn:
    image_source = uploaded_file if uploaded_file is not None else camera_file
    assert image_source is not None  # only reachable if button enabled

    try:
        st.spinner("Running smoke detection...")
        image_bytes = image_source.read()
        result = _run_prediction(image_bytes)

        label_name = result["label_name"]
        smoke_pct = float(result["smoke_pct"])

        style = _label_style(label_name)
        result_msg = f"{style['prefix']}  |  Smoke density: {smoke_pct:.2f}%"

        if style["kind"] == "success":
            result_box.success(result_msg)
        elif style["kind"] == "warning":
            result_box.warning(result_msg)
        elif style["kind"] == "error":
            result_box.error(result_msg)
        else:
            result_box.info(result_msg)

        # Show transmission-map preview (grayscale)
        if result.get("t_est_preview_b64"):
            t_img = _b64png_to_pil(result["t_est_preview_b64"])
            preview_box.image(t_img, caption="Estimated transmission map (preview)", use_column_width=True)

    except Exception as e:  # noqa: BLE001
        result_box.error(f"Prediction failed: {e}")


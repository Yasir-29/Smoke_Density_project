from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from feature_extraction import extract_features, label_to_name
from preprocessing import preprocess_image


@lru_cache(maxsize=1)
def load_classifier(model_path: str) -> Any:
    """Load the trained sklearn model once (cached)."""
    return joblib.load(model_path)


def _b64_png_from_gray01(img01: np.ndarray) -> str:
    """Convert a 0..1 float grayscale image to base64-encoded PNG."""
    img_u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("Failed to encode PNG preview.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def predict_smoke_from_bgr(
    clf: Any,
    img_bgr: np.ndarray,
) -> dict[str, Any]:
    """
    Predict smoke class and smoke percentage from a BGR image.

    Returns JSON-serializable dict for the webpage.
    """
    img = preprocess_image(img_bgr, size=(256, 256))
    feats, t_est = extract_features(img)
    X = np.array([feats], dtype=np.float32)

    pred_label = int(clf.predict(X)[0])
    smoke_density = 1.0 - float(np.mean(t_est))
    smoke_pct = 100.0 * smoke_density

    return {
        "label_id": pred_label,
        "label_name": label_to_name(pred_label),
        "smoke_pct": round(smoke_pct, 2),
        "smoke_density": round(smoke_density, 6),
        "t_est_preview_b64": _b64_png_from_gray01(t_est),
    }


def predict_smoke_from_bytes(
    model_path: Path,
    image_bytes: bytes,
) -> dict[str, Any]:
    """Predict smoke from raw uploaded bytes (JPEG/PNG/etc.)."""
    clf = load_classifier(str(model_path))

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image bytes.")

    return predict_smoke_from_bgr(clf, img_bgr)


import cv2
import os
import numpy as np

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_image(img: np.ndarray, size=(256, 256)) -> np.ndarray:
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

# -----------------------------
# LOAD DATASET FUNCTION
# -----------------------------
def _haze_to_trans_filename(haze_filename: str) -> str | None:
    """
    RESIDE ITS typical haze name: 10001_01_0.9797.png
    Ground-truth transmission name: 10001_01.png (first two underscore parts).
    """
    stem, _ext = os.path.splitext(haze_filename)
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    return f"{parts[0]}_{parts[1]}.png"


def load_dataset(
    haze_path: str,
    trans_path: str,
    limit: int | None = 500,
    size=(256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    haze_images: list[np.ndarray] = []
    trans_images: list[np.ndarray] = []

    haze_files = sorted(
        f for f in os.listdir(haze_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for haze_file in haze_files:
        if limit is not None and len(haze_images) >= limit:
            break

        trans_file = _haze_to_trans_filename(haze_file)
        if trans_file is None:
            continue

        haze_img_path = os.path.join(haze_path, haze_file)
        trans_img_path = os.path.join(trans_path, trans_file)

        haze = cv2.imread(haze_img_path, cv2.IMREAD_COLOR)
        trans = cv2.imread(trans_img_path, cv2.IMREAD_GRAYSCALE)
        if haze is None or trans is None:
            continue

        haze = preprocess_image(haze, size=size)
        trans = preprocess_image(trans, size=size)
        if trans.ndim == 3:
            trans = trans[:, :, 0]

        haze_images.append(haze)
        trans_images.append(trans)

    return np.stack(haze_images, axis=0), np.stack(trans_images, axis=0)


def load_dataset_with_ids(
    haze_path: str,
    trans_path: str,
    limit: int | None = 500,
    size=(256, 256),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as load_dataset(), but also returns a per-sample scene id.
    For RESIDE ITS, this is the first two underscore parts (e.g., 10001_01).
    """
    haze_images: list[np.ndarray] = []
    trans_images: list[np.ndarray] = []
    scene_ids: list[str] = []

    haze_files = sorted(
        f for f in os.listdir(haze_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for haze_file in haze_files:
        if limit is not None and len(haze_images) >= limit:
            break

        stem, _ext = os.path.splitext(haze_file)
        parts = stem.split("_")
        if len(parts) < 2:
            continue

        scene_id = f"{parts[0]}_{parts[1]}"
        trans_file = f"{scene_id}.png"

        haze_img_path = os.path.join(haze_path, haze_file)
        trans_img_path = os.path.join(trans_path, trans_file)

        haze = cv2.imread(haze_img_path, cv2.IMREAD_COLOR)
        trans = cv2.imread(trans_img_path, cv2.IMREAD_GRAYSCALE)
        if haze is None or trans is None:
            continue

        haze = preprocess_image(haze, size=size)
        trans = preprocess_image(trans, size=size)
        if trans.ndim == 3:
            trans = trans[:, :, 0]

        haze_images.append(haze)
        trans_images.append(trans)
        scene_ids.append(scene_id)

    return np.stack(haze_images, axis=0), np.stack(trans_images, axis=0), np.array(scene_ids)
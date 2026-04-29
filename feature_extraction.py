import cv2
import numpy as np

# -----------------------------
# DARK CHANNEL
# -----------------------------
def dark_channel(image, size=15):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("dark_channel expects an HxWx3 image in range [0,1].")
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

# -----------------------------
# ATMOSPHERIC LIGHT
# -----------------------------
def atmospheric_light(image, dark):
    flat = dark.ravel()
    k = min(1000, flat.size)
    indices = flat.argsort()[-k:]
    brightest = image.reshape(-1, 3)[indices]
    A = np.max(brightest, axis=0)
    return np.maximum(A, 1e-6)

# -----------------------------
# TRANSMISSION ESTIMATION
# -----------------------------
def transmission(image, A, omega=0.95):
    norm = image / A
    dark = dark_channel(norm)
    t = 1 - omega * dark
    return np.clip(t, 0.0, 1.0)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(image):
    dark = dark_channel(image)
    A = atmospheric_light(image, dark)
    t_est = transmission(image, A)

    features = [
        np.mean(t_est),
        np.std(t_est),
        np.min(t_est),
        np.max(t_est)
    ]

    return features, t_est

# -----------------------------
# LABEL GENERATION (FROM TRUE TRANS)
# -----------------------------
def generate_label(trans):
    if trans.ndim == 3:
        trans = trans[:, :, 0]
    trans = np.clip(trans, 0.0, 1.0)
    density = 1 - float(np.mean(trans))

    # 0: Good, 1: Moderate, 2: Hazardous
    if density < 0.2:
        return 0
    if density < 0.5:
        return 1
    return 2


def label_to_name(label: int) -> str:
    return {0: "Good", 1: "Moderate", 2: "Hazardous"}.get(int(label), "Unknown")
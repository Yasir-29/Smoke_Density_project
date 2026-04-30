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
    """
    Compute a richer set of DCP-based features from the image.

    All values are derived from the estimated transmission map and the
    dark channel, which correlate with smoke / haze density.
    """
    dark = dark_channel(image)
    A = atmospheric_light(image, dark)
    t_est = transmission(image, A)

    # Flatten for easy statistics
    t_flat = t_est.ravel()
    d_flat = dark.ravel()

    # Transmission statistics
    t_mean = float(np.mean(t_flat))
    t_std = float(np.std(t_flat))
    t_min = float(np.min(t_flat))
    t_max = float(np.max(t_flat))
    t_p10 = float(np.percentile(t_flat, 10))
    t_p25 = float(np.percentile(t_flat, 25))
    t_p50 = float(np.percentile(t_flat, 50))
    t_p75 = float(np.percentile(t_flat, 75))
    t_p90 = float(np.percentile(t_flat, 90))

    # Dark-channel statistics
    d_mean = float(np.mean(d_flat))
    d_std = float(np.std(d_flat))
    d_min = float(np.min(d_flat))
    d_max = float(np.max(d_flat))

    # Atmospheric light magnitude (per-channel + norm)
    A_r, A_g, A_b = [float(v) for v in A]
    A_norm = float(np.linalg.norm(A))

    features = [
        # Transmission stats
        t_mean,
        t_std,
        t_min,
        t_max,
        t_p10,
        t_p25,
        t_p50,
        t_p75,
        t_p90,
        # Dark channel stats
        d_mean,
        d_std,
        d_min,
        d_max,
        # Atmospheric light
        A_r,
        A_g,
        A_b,
        A_norm,
    ]

    return features, t_est

# -----------------------------
# LABEL GENERATION (FROM TRUE TRANS)
# -----------------------------
def generate_label(trans):
    if trans.ndim == 3:
        trans = trans[:, :, 0]
    trans = np.clip(trans, 0.0, 1.0)
    
    # Calculate density based on the 25% thickest smoke
    t_flat = trans.ravel()
    k = max(1, int(t_flat.size * 0.25))
    lowest_t = np.partition(t_flat, k)[:k]
    density = 1.0 - float(np.mean(lowest_t))

    # 0: Low, 1: Moderate, 2: High
    if density < 0.25:
        return 0
    if density <= 0.60:
        return 1
    return 2


def label_to_name(label: int) -> str:
    return {0: "Low", 1: "Moderate", 2: "High"}.get(int(label), "Unknown")
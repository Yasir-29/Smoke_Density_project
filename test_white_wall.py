import cv2
import numpy as np
import joblib
from feature_extraction import extract_features

clf = joblib.load('outputs/model.joblib')

# Create a dummy white wall image
img_bgr = np.ones((256, 256, 3), dtype=np.uint8) * 220
# Add a little noise so it's not perfectly zero variance
noise = np.random.normal(0, 2, img_bgr.shape).astype(np.uint8)
img_bgr = cv2.add(img_bgr, noise)

img_norm = img_bgr / 255.0
feats, t_est = extract_features(img_norm)
X = np.array([feats], dtype=np.float32)

if hasattr(clf, "n_features_in_"):
    X = X[:, :int(clf.n_features_in_)]

pred = clf.predict(X)[0]
probs = clf.predict_proba(X)[0]
print(f"White wall predicted label: {pred}")
print(f"Probabilities: {probs}")

t_flat = t_est.ravel()
k = max(1, int(t_flat.size * 0.25))
lowest_t = np.partition(t_flat, k)[:k]
smoke_density = 1.0 - float(np.mean(lowest_t))
print(f"Transmission smoke density: {smoke_density * 100:.2f}%")

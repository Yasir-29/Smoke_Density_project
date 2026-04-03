import cv2
import numpy as np

def dark_channel(image, size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def atmospheric_light(image, dark):
    flat = dark.ravel()
    indices = flat.argsort()[-1000:]
    brightest = image.reshape(-1, 3)[indices]
    return np.max(brightest, axis=0)

def transmission(image, A, omega=0.95):
    norm = image / A
    dark = dark_channel(norm)
    return 1 - omega * dark

# Load image
img = cv2.imread("dataset/indoor/hazy/1400_1.png")

img = cv2.resize(img, (256, 256)) / 255.0

dark = dark_channel(img)
A = atmospheric_light(img, dark)
t = transmission(img, A)

density = 1 - t.mean()

print("Smoke Density:", density)
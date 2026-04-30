import numpy as np
import cv2

# Simulate Capitol image: top half is bright smooth sky (val ~200), bottom is dark trees/buildings (val ~50)
img = np.zeros((256, 256), dtype=np.uint8)
img[:128, :] = 200
img[128:, :] = 50

# Add some noise
noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
img = cv2.add(img, noise)

mean_val, std_val = cv2.meanStdDev(img)
print(f"Simulated Capitol - Brightness: {mean_val[0][0]:.2f}, Std: {std_val[0][0]:.2f}")

# Simulate white wall: uniformly 150
img2 = np.ones((256, 256), dtype=np.uint8) * 150
noise2 = np.random.normal(0, 15, img2.shape).astype(np.uint8) # heavy phone noise
img2 = cv2.add(img2, noise2)

mean_val2, std_val2 = cv2.meanStdDev(img2)
print(f"Simulated White Wall - Brightness: {mean_val2[0][0]:.2f}, Std: {std_val2[0][0]:.2f}")

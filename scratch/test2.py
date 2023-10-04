import numpy as np

im = np.random.randn(100, 100) * 0.1  # random image

mask = np.zeros((100, 100))
mask[:, 30] = 1  # white square in black background
# im = mask + np.random.randn(10, 10) * 0.1  # random image
masked = mask

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(im, interpolation="none")
plt.subplot(1, 2, 2)
plt.imshow(masked, "gray", interpolation="none")
plt.imshow(im, interpolation="none", alpha=1.0 * (masked == 0))

plt.show()

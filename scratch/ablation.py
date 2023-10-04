import matplotlib.pyplot as plt
import numpy as np


x = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
y = np.array(
    [
        0.8182,
        0.9311,
        0.9356,
        0.9360,
        0.9582,
        0.9524,
        0.9520,
        0.9560,
        0.9480,
    ]
)

plt.plot(x, y)
plt.ylim([0, 1])
plt.show()

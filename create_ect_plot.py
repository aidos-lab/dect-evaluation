import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2 * np.pi * x)
y2 = 0.8 * np.sin(4 * np.pi * x)

fig, ax = plt.subplots()
ax.fill_between(x, y1, -1, color="gray", alpha=0.2)
ax.fill_between(x, y2, -1, color="gray", alpha=0.2)
ax.set_axis_off()
fig.tight_layout()
plt.savefig("test.png", bbox_inches="tight")
plt.show()

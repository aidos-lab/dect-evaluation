import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D


def waterfall_plot(fig, ax, X, Y, Z):
    """
    Make a waterfall plot
    Input:
        fig,ax : matplotlib figure and axes to populate
        Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
        X,Y : n,m array
    """
    # Set normalization to the same values for all plots
    norm = plt.Normalize(Z.min().min(), Z.max().max())
    # Check sizes to loop always over the smallest dimension
    n, m = Z.shape
    if n > m:
        X = X.T
        Y = Y.T
        Z = Z.T
        m, n = n, m

    for j in range(n):
        # reshape the X,Z into pairs
        points = np.array([X[j, :], Z[j, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="plasma", norm=norm)
        # Set the values used for colormapping
        lc.set_array((Z[j, 1:] + Z[j, :-1]) / 2)
        lc.set_linewidth(
            2
        )  # set linewidth a little larger to see properly the colormap variation
        line = ax.add_collection3d(
            lc, zs=(Y[j, 1:] + Y[j, :-1]) / 2, zdir="y"
        )  # add line to axes

    fig.colorbar(
        lc
    )  # add colorbar, as the normalization is the same for all, it doesent matter which of the lc objects we use


ect = np.load("./ect.npy")
print(ect.shape)
# Generate data
x = np.linspace(-2, 2, 512)
y = np.linspace(-2, 2, 512)
X, Y = np.meshgrid(x, y)
Z = ect


import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="plasma")

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

plt.show()


plt.plot_surface(X, Y, Z)

# # Generate waterfall plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# waterfall_plot(fig, ax, X, Y, Z)
# ax.set_xlabel("X")
# ax.set_xlim3d(-2, 2)
# ax.set_ylabel("Y")
# ax.set_ylim3d(-2, 2)
# ax.set_zlabel("Z")
# ax.set_zlim3d(-1, 1000)
# plt.show()

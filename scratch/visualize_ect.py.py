from models.layers.fixedlayers import EctPointsLayer, EctEdgesLayer, EctFacesLayer
from models.base_model import ECTModelConfig
from dataclasses import dataclass
import torch
import geotorch
import numpy as np
import sys
from torch_geometric.data import Batch, Data

from matplotlib import pyplot as plt


np.set_printoptions(threshold=sys.maxsize)

"""
This is test script to verify that the ect layer works as intended. 
We initialize the ECTLayer and create a simple point cloud and compute the 
ect. 
Afterwards we visualize the ECT.
"""


@dataclass
class ect:
    num_thetas = 64
    num_features = 2
    R = 1.1
    bump_steps = 64


# Basic dataset with three points,three edges and one face.
data = Data(
    x=torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]),
    edge_index=torch.tensor([[0, 1, 2, 2], [2, 2, 0, 1]], dtype=torch.long),
    face=torch.tensor([[0], [1], [2]], dtype=torch.long),
)
batch = Batch.from_data_list([data])
print(batch)


# Initialize from fixed layers
pointslayer = EctPointsLayer(ect())
# edgeslayer = EctEdgesLayer(ect())
# faceslayer = EctFacesLayer(ect())

# # Plot angles and graph.
# angles = pointslayer.v.T.cpu().detach().numpy()
# plt.scatter(angles[:, 0], angles[:, 1])
# plt.scatter(batch.x[:, 0], batch.x[:, 1])
# plt.show()


# Compute ect
points_out = pointslayer(batch.cuda()).squeeze().cpu().numpy()
# edges_out = edgeslayer(batch.cuda())
# faces_out = faceslayer(batch.cuda())
# print(points_out)
# print(edges_out)
# print(faces_out)

print(points_out.shape)


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


# Generate data
x = np.linspace(-1, 1, 64)
y = np.linspace(0, 2 * np.pi, 64)
X, Y = np.meshgrid(x, y)


# Generate waterfall plot
fig = plt.figure(figsize=(30, 10))

# Plot angles and graph.
angles = pointslayer.v.T.cpu().detach().numpy()
ax = fig.add_subplot(131)
ax.scatter(angles[:, 0], angles[:, 1])
ax.scatter(batch.x[:, 0].cpu().numpy(), batch.x[:, 1].cpu().numpy())


ax = fig.add_subplot(132, projection="3d")
waterfall_plot(fig, ax, X, Y, points_out.T)
ax.set_xlabel("Filter")
ax.set_xlim3d(-1, 1)
ax.set_ylabel("Theta")
ax.set_ylim3d(0, 2 * np.pi)
ax.set_zlabel("ECC")
ax.set_zlim3d(-5, 5)

ax2 = fig.add_subplot(133)
# add colorbar, as the normalization is the same for all, it doesent matter which of the lc objects we use

ax2.imshow(points_out, cmap="plasma", clim=[0, 3])

plt.show()

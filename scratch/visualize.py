from models.layers.fixedlayers import EctPointsLayer as ECTPointsLayerTruth
from models.layers.layers import EctPointsLayer as ECTPointsLayerPred
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


def sample_from_sphere(n=200, d=2, r=0.2, noise=None, ambient=None, seed=None):
    rng = np.random.default_rng(seed)
    # np.random.standard_normal(size=...)
    data = rng.standard_normal((n, d + 1))

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None]) + 0.5

    if noise:
        data += noise * rng.standard_normal(data.shape)

    return torch.as_tensor(data, dtype=torch.float32)


@dataclass
class ect:
    num_thetas = 512
    num_features = 2
    R = 1.1
    bump_steps = 512


# Basic dataset with three points,three edges and one face.
data = Data(
    x=sample_from_sphere(d=1, noise=0.01),
)
batch = Batch.from_data_list([data])

device = "cuda:0"
# # Initialize from fixed layers
ect_ground_truth = ECTPointsLayerTruth(ect()).to(device)
ground_truth = ect_ground_truth(batch.to(device=device)).cpu().squeeze().numpy()
# plt.imshow(ground_truth.squeeze().cpu().detach().numpy())
# plt.show()

np.save("ect", ground_truth)

plt.imsave("groundthruth.jpg", ground_truth)


index = 64

fig = plt.figure(figsize=(10, 10))
plt.tight_layout(pad=0.0)
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[:, 1])
ax2 = fig.add_subplot(gs[0, 0])
ax3 = fig.add_subplot(gs[1, 0])


mask = np.zeros((256, 266, 3))
mask[:, index, 0] = 1
ax1.imshow(mask, interpolation="none")
ax1.imshow(ground_truth, interpolation="none", alpha=1.0 * (mask[:, :, 0] == 0))
ax1.axis("off")


angles = ect_ground_truth.v.T.cpu().detach().numpy()
batch = batch.cpu()
circle = plt.Circle((0, 0), 1, color="b", fill=False)
ax2.add_patch(circle)
ax2.scatter(batch.x[:, 0], batch.x[:, 1])
ax2.scatter(angles[index, 0], angles[index, 1], c="r")
ax2.set_xlim([-1.1, 1.1])
ax2.set_ylim([-1.1, 1.1])
ax2.axis("off")

ax3.plot(ground_truth[:, index])
ax3.axis("off")

plt.savefig("./ect_overview.jpg", bbox_inches="tight")
plt.show()


plt.imshow(ground_truth)
plt.axis("off")
plt.savefig("./ect_ground.svg", bbox_inches="tight")
plt.show()

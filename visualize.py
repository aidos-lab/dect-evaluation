import torch
from models.layers.layers import EctPointsLayer
from datasets.tu import TUNCI109Config, TUDataModule
from models.base_model import ECTModelConfig
from datasets.modelnet import ModelNetPointsDataModule, ModelNetDataModuleConfig
import numpy as np

dataset = ModelNetPointsDataModule(
    ModelNetDataModuleConfig(
        root="./data/modelnet_aligned_40_100",
        module="datasets.modelnet",
        name="40",
        samplepoints=100,
    )
)


for batch in dataset.train_dataloader():
    break


print(batch.x.norm(dim=-1).max())

layer = EctPointsLayer(
    ECTModelConfig(module="", num_features=3, bump_steps=36, num_thetas=36)
).to("cuda:0")

# print(batch.x)
# print(batch.x.shape)

import matplotlib.pyplot as plt

idx = 3

res = layer(batch.to("cuda:0"))
print(res.shape)

img = res[idx].reshape(36, 36).detach().cpu().numpy()


# print("MAX", np.max(img))

print(res[idx])
plt.imshow(img)
plt.show()

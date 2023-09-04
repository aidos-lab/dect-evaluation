import torch
from models.layers.layers import EctPointsLayer
from datasets.tu import TUNCI109Config, TUDataModule
from models.base_model import ECTModelConfig
from datasets.modelnet import ModelNetPointsDataModule, ModelNetDataModuleConfig
import numpy as np

dataset = ModelNetPointsDataModule(
    ModelNetDataModuleConfig(
        root="./data/modelnet_aligned_40_1000",
        module="datasets.modelnet",
        name="40",
        samplepoints=1000,
    )
)


for batch in dataset.train_dataloader():
    break

layer = EctPointsLayer(
    ECTModelConfig(module="", num_features=3, bump_steps=36, num_thetas=36)
).to("cuda:0")

print(batch.x)
print(batch.x.shape)

res = layer(batch.to("cuda:0"))
print(res.shape)

img = res[0].reshape(36, 36).detach().cpu().numpy()

import matplotlib.pyplot as plt

print(np.max(img))

print(res[0])
plt.imshow(img)
plt.show()

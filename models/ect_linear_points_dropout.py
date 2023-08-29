import torch
import geotorch
import torch.nn as nn
from models.base_model import BaseModel
from dataclasses import dataclass

from models.layers.layers import EctPointsLayer


class ECTLinearPointsModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ectlayer = EctPointsLayer(config)
        geotorch.constraints.sphere(self.ectlayer, "v")
        self.linear1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(config.num_thetas * config.bump_steps, config.hidden),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(config.hidden, config.hidden), nn.ReLU()
        )
        self.linear3 = nn.Sequential(nn.Linear(config.hidden, config.num_classes))

    def forward(self, batch):
        x = self.ectlayer(batch).reshape(
            -1, self.config.num_thetas * self.config.bump_steps
        )
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


from loaders.factory import register


def initialize():
    register("model", ECTLinearPointsModel)

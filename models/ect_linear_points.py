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
        self.linear1 = torch.nn.Linear(
            config.num_thetas * config.bump_steps, config.hidden
        )
        self.linear2 = torch.nn.Linear(config.hidden, config.hidden)
        self.linear3 = nn.Linear(config.hidden, config.num_classes)

    def forward(self, batch):
        x = (
            self.ectlayer(batch).reshape(
                -1, self.config.num_thetas * self.config.bump_steps
            )
            / 1000
        )
        """ x = x / x.max(dim=1)[0].unsqueeze(1) """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x


from loaders.factory import register


def initialize():
    register("model", ECTLinearPointsModel)

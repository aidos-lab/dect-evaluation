import torch
import geotorch
import torch
import torch.nn as nn
from models.base_model import BaseModel
from dataclasses import dataclass

from models.layers.layers import EctFacesLayer
from loaders.factory import register

#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯

#  ╭──────────────────────────────────────────────────────────╮
#  │ These two models only take the point cloud               │
#  │  structure into account.                                 │
#  ╰──────────────────────────────────────────────────────────╯


class ECTLinearFacesModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ectlayer = EctFacesLayer(config)
        geotorch.constraints.sphere(self.ectlayer, "v")
        self.linear1 = torch.nn.Linear(
            config.num_thetas * config.bump_steps, config.hidden
        )
        self.linear2 = torch.nn.Linear(config.hidden, 100)
        self.linear3 = nn.Linear(100, config.num_classes)

    def forward(self, batch):
        x = self.ectlayer(batch).reshape(
            -1, self.config.num_thetas * self.config.bump_steps
        )
        """ x = x / torch.abs(x).max(dim=1)[0].unsqueeze(1) """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x


def initialize():
    register("model", ECTLinearFacesModel)
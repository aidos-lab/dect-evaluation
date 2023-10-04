import torch
import geotorch
import torch.nn as nn
import functools
import operator


from models.layers.layers import EctLayer
from models.config import ModelConfig
from loaders.factory import register
from models.base_model import BaseModel

from dataclasses import dataclass

#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯

#  ╭──────────────────────────────────────────────────────────╮
#  │ These two models only take the point cloud               │
#  │  structure into account.                                 │
#  ╰──────────────────────────────────────────────────────────╯


class ECTCNNPointsModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.ectlayer = EctLayer(config.ectconfig, ecc_type="points")
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.MaxPool2d(2),
        )
        num_features = functools.reduce(
            operator.mul,
            list(self.conv1(torch.rand(1, config.bump_steps, config.num_thetas)).shape),
        )
        self.linear1 = nn.Linear(num_features, config.hidden)
        self.linear2 = nn.Linear(config.hidden, config.hidden)
        self.linear3 = nn.Linear(config.hidden, config.num_classes)

    def forward(self, batch):
        x = self.ectlayer(batch)
        x = x.unsqueeze(1)
        """ x = x / torch.max(torch.abs(x)) """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        return x


def initialize():
    register("model", ECTCNNPointsModel)

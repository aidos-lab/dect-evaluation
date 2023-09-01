import torch
import geotorch
import torch
import torch.nn as nn
import functools
import operator
from models.base_model import BaseModel

from models.layers.layers import EctPointsLayer, EctEdgesLayer

from dataclasses import dataclass

#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯

#  ╭──────────────────────────────────────────────────────────╮
#  │ These two models only take the point cloud               │
#  │  structure into account.                                 │
#  ╰──────────────────────────────────────────────────────────╯


class ECTCNNPointsModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ectlayer = EctPointsLayer(config)
        geotorch.constraints.sphere(self.ectlayer, "v")

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            # nn.BatchNorm1d(30),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3),
            # nn.BatchNorm2d(13),
            nn.MaxPool2d(2),
        )
        num_features = functools.reduce(
            operator.mul,
            list(self.conv1(torch.rand(1, config.bump_steps, config.num_thetas)).shape),
        )
        self.linear1 = nn.Linear(num_features, config.hidden)
        self.linear2 = nn.Linear(config.hidden, config.hidden // 2)
        self.linear3 = nn.Linear(config.hidden // 2, config.num_classes)
        # self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()
        # self.layer_norm = nn.LayerNorm([32, 32])

    def forward(self, batch):
        x = self.ectlayer(batch)
        # x = self.layer_norm(x)
        x = x.unsqueeze(1)
        """ x = x / torch.max(torch.abs(x)) """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.dropout3(x)
        x = self.linear3(x)
        return x


from loaders.factory import register


def initialize():
    register("model", ECTCNNPointsModel)
    # register("model", VGG16)

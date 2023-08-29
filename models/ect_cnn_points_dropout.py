import torch
import geotorch
import torch
import torch.nn as nn
import functools
import operator
from models.base_model import BaseModel

from models.layers.layers import EctPointsLayer

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
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.linear1 = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, config.hidden), nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(config.hidden, config.hidden), nn.ReLU()
        )
        self.linear3 = nn.Sequential(nn.Linear(config.hidden, config.num_classes))

        # self.linear1 = nn.Linear(num_features, config.hidden)
        # self.linear2 = nn.Linear(config.hidden, 400)
        # self.linear3 = nn.Linear(400, config.num_classes)

    def forward(self, batch):
        x = self.ectlayer(batch)
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


from loaders.factory import register


def initialize():
    register("model", ECTCNNPointsModel)

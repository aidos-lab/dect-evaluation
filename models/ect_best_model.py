import geotorch
import torch
import torch.nn as nn
import functools
import operator
from models.base_model import BaseModel
from models.layers.layers import EctPointsLayer
from loaders.factory import register


class ECTCNNPointsModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ectlayer = EctPointsLayer(config)
        geotorch.constraints.sphere(self.ectlayer, "v")

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3), stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2), stride=2),
        )
        num_features = functools.reduce(
            operator.mul,
            list(
                self.conv1(torch.rand(1, 1, config.bump_steps, config.num_thetas)).shape
            ),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(num_features, config.num_classes),
        )

    def forward(self, batch):
        x = self.ectlayer(batch).unsqueeze(1)
        x /= 50
        x -= 1.0
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def initialize():
    register("model", ECTCNNPointsModel)

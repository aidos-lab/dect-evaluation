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
            nn.Conv2d(1, 6, kernel_size=(5), stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2), stride=2),
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=(5), stride=1, padding=0),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2), stride=2),
        # )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(),
        # )

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=3),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 16, kernel_size=3),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        # )
        num_features = functools.reduce(
            operator.mul,
            list(
                # self.conv3(
                self.conv2(
                    self.conv1(
                        torch.rand(1, 1, config.bump_steps, config.num_thetas)
                        # )
                    )
                ).shape
            ),
        )
        # self.layernorm = nn.LayerNorm([self.config.num_thetas, self.config.bump_steps])
        self.linear = nn.Sequential(
            nn.Linear(num_features, config.hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(config.hidden, config.hidden // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(config.hidden // 2, config.num_classes),
        )

    def forward(self, batch):
        x = self.ectlayer(batch).unsqueeze(1)
        # x = self.layernorm(x)
        x /= 100
        x -= 0.5
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def initialize():
    register("model", ECTCNNPointsModel)

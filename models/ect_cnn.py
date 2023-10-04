import torch
import torch.nn as nn
import functools
import operator


from models.layers.layers import EctLayer
from models.config import ModelConfig
from loaders.factory import register
from models.base_model import BaseModel


class EctCnnModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.ectlayer = EctLayer(config.ectconfig)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.MaxPool2d(2),
        )
        num_features = functools.reduce(
            operator.mul,
            list(
                self.conv(
                    torch.rand(
                        1, config.ectconfig.bump_steps, config.ectconfig.num_thetas
                    )
                ).shape
            ),
        )

        self.linear = nn.Sequential(
            nn.Linear(num_features, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.num_classes),
        )

    def forward(self, batch):
        x = self.ectlayer(batch).unsqueeze(1)
        x = self.conv(x).view(x.size(0), -1)
        x = self.linear(x)
        return x


def initialize():
    register("model", EctCnnModel)

import torch
import torch.nn as nn
from models.base_model import BaseModel


from models.layers.layers_wect import WectLayer
from models.config import EctConfig


class EctLinearModel(BaseModel):
    def __init__(self, config: EctConfig):
        super().__init__(config)
        self.ectlayer = WectLayer(config.ectconfig)

        self.linear = nn.Sequential(
            nn.Linear(self.config.num_thetas * self.config.bump_steps, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.num_classes),
        )

    def forward(self, batch):
        x = self.ectlayer(batch).reshape(
            -1, self.config.num_thetas * self.config.bump_steps
        )
        x = self.linear(x)
        return x


from loaders.factory import register


def initialize():
    register("model", EctLinearModel)

import torch.nn as nn
from models.base_model import BaseModel

from models.layers.layers_wect import WectLayer
from models.config import EctConfig


class WectLinearModel(BaseModel):
    def __init__(self, config: EctConfig):
        super().__init__(config)
        
        self.ectlayer = WectLayer(config.ectconfig)
        print("ran THIS")
        self.linear = nn.Sequential(
            nn.Linear(self.config.ectconfig.num_thetas * self.config.ectconfig.bump_steps, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.num_classes),
        )

    def forward(self, batch):
        x = self.ectlayer(batch).reshape(
            -1, self.config.ectconfig.num_thetas * self.config.ectconfig.bump_steps
        )
        x = self.linear(x)
        return x


from loaders.factory import register


def initialize():
    register("model", WectLinearModel)

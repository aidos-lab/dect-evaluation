import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
    module: str
    num_features: int = 3
    bump_steps: int = 32 
    num_thetas: int = 32 
    num_classes: int = 10
    hidden: int = 100 
    R: float = 1.1
    scale: float = 200


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()


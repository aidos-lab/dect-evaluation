from torch import nn
from dataclasses import dataclass


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


@dataclass(frozen=True)
class ECTModelConfig:
    module: str
    num_thetas: int = 32
    bump_steps: int = 32
    batch_size: int = 128
    R: float = 1.1
    num_features: int = 3
    num_classes: int = 10
    hidden: int = 50

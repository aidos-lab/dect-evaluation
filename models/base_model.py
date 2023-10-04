from torch import nn
from dataclasses import dataclass


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
<<<<<<< HEAD


@dataclass(frozen=True)
class ECTModelConfig:
    module: str
    num_phis: int = 12
    num_thetas: int = 13
    bump_steps: int = 14
    batch_size: int = 128
    R: float = 1.1
    num_features: int = 3
    num_classes: int = 10
    hidden: int = 50
=======
        self.loss_fn = nn.CrossEntropyLoss()
>>>>>>> main

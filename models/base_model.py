from torch import nn
from dataclasses import dataclass


class BaseModel(nn.Module):
    """
    This is an abstract base model for the model class.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

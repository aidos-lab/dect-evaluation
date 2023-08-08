import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

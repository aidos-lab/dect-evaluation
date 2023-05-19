
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from models.base_model import BaseModel

class NN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.lr = config.learning_rate
        self.fc1 = nn.Linear(config.input_size, 1_00)
        self.fc2 = nn.Linear(1_00, config.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=config.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



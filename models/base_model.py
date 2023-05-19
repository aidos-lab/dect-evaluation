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


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.config.num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.config.num_classes)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)

        # self.log_dict(
        #     {
        #         "train_loss": loss,
        #     },
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     batch_size=self.config.batch_size
        # )
        
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        # self.log("val_loss", loss, batch_size=self.config.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        # self.log("test_loss", loss, batch_size=self.config.batch_size)
        return loss

    def _common_step(self, batch, batch_idx):
        y = batch.y
        scores = self.forward(batch)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.lr)

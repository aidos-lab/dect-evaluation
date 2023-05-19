from datasets.base_dataset import DataModule
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from types import SimpleNamespace
from torch.utils.data import random_split
import torch
from torchvision.datasets import MNIST
from torch_geometric.datasets import GNNBenchmarkDataset
import torchvision.transforms as transforms
import torch_geometric
from types import SimpleNamespace
from logs import log_msg
from omegaconf import OmegaConf
#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯

class ThresholdTransform(object):
  def __call__(self, data):
    x = torch.hstack([data.pos,data.x]) 
    x -= torch.tensor(0.5)
    new_data = torch_geometric.data.Data(x=x, edge_index=data.edge_index,y=data.y)
    return new_data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

class GNNBenchmarkDataModule(DataModule):
    def __init__(self,config):
        super().__init__(config.root,config.batch_size,config.num_workers)
        self.config = config
        self.prepare_data()
        self.setup()

    
    def prepare_data(self):
        pre_transform = transforms.Compose([ThresholdTransform()])
        GNNBenchmarkDataset(
                name = self.config.name,
                root = self.config.root,
                pre_transform=pre_transform,
                split = "train"
                )
        GNNBenchmarkDataset(
                name = self.config.name,
                root = self.config.root,
                pre_transform=pre_transform,
                split = "test"
                )
        GNNBenchmarkDataset(
                name = self.config.name,
                root = self.config.root,
                pre_transform=pre_transform,
                split = "val"
                )

    def setup(self):
        pre_transform = transforms.Compose([ThresholdTransform()])
        self.train_ds = GNNBenchmarkDataset(
                name = self.config.name,
                root = self.config.root,
                pre_transform=pre_transform,
                split = "train"
                )
        self.test_ds = GNNBenchmarkDataset(
                name = self.config.name,
                root = self.config.root,
                pre_transform=pre_transform,
                split = "test"
                )
        self.val_ds = GNNBenchmarkDataset(
                name = self.config.name,
                root = self.config.root,
                pre_transform=pre_transform,
                split = "val"
                )


from datasets.base_dataset import DataModule
import torch
import torchvision.transforms as transforms
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
import torchvision.transforms as transforms
import torch_geometric

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
        self.config = config
        super().__init__(config.root,config.batch_size,config.num_workers)

    
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


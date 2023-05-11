from datasets.base_dataset import BaseDataset
import torch
from torchvision.datasets import MNIST
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import TUDataset
import torchvision.transforms as transforms
import torch_geometric
from types import SimpleNamespace
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

class GNN_MNIST(BaseDataset):
    def __init__(self,config):
        config.pre_transform = transforms.Compose([ThresholdTransform()])
        config.name = "MNIST"
        super().__init__(dataset=GNNBenchmarkDataset,config=config)





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

class GNN_MNIST():
    def __init__(self,config):
        pre_transform = transforms.Compose([ThresholdTransform()])
        self.dataset = GNNBenchmarkDataset(**vars(config)|{"pre_transform":pre_transform})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)




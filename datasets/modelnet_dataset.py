from datasets.base_dataset import BaseDataset
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.datasets import TUDataset
import torchvision.transforms as transforms
import torch_geometric
from types import SimpleNamespace
#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯


#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

class GNN_ModelNet(BaseDataset):
    def __init__(self,config):
        config.pre_transform = transforms.Compose([torch_geometric.transforms.SamplePoints(100)])
        super().__init__(dataset=ModelNet,config=config)





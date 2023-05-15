from datasets.base_dataset import BaseDataset
import torch
from torch_geometric.datasets import ModelNet
import torchvision.transforms as transforms
import torch_geometric
from types import SimpleNamespace
from pretty_simple_namespace import pprint
#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯

class ModelNetTransform(object):
  def __call__(self, data):
    data.x = data.pos
    return data

class CenterTransform(object):
  def __call__(self, data):
    data.x -= data.x.mean()
    data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
    return data

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

class GNN_ModelNet100():
    def __init__(self,config):
        pre_transform = transforms.Compose([torch_geometric.transforms.SamplePoints(100),
                                                   ModelNetTransform(),
                                                   CenterTransform()])
        print(vars(config))
        self.dataset = ModelNet(**vars(config)|{"pre_transform":pre_transform})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


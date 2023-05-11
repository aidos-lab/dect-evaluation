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

class GNN_ModelNet100(BaseDataset):
    def __init__(self,config):
        pprint(config)        
        config.pre_transform = transforms.Compose([torch_geometric.transforms.SamplePoints(100),
                                                   ModelNetTransform(),
                                                   CenterTransform()])
        super().__init__(dataset=ModelNet,config=config)



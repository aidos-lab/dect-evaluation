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

"""
Add all the required transforms in this section, or use imports.
"""

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

"""
Define the dataset classes, provide dataset/dataloader parameters 
in the config file or overwrite them in the class definition.
"""

class TU_DATASET(BaseDataset):
    def __init__(self,config):
        """
        This is the "flexible" base class for testing, for the experiments we 
        fix a set of parameters in the config file and run the experiment.
        """
        config.name = "Letter-high"
        config.root = "./data"
        #config.pre_transform = transforms.Compose([ThresholdTransform()])
        super().__init__(dataset=TUDataset,config=config)


class PROTEINS_DATASET(BaseDataset):
    def __init__(self,config):
        #config.pre_transform = transforms.Compose([ThresholdTransform()])
        config.root = "./data"
        config.name="PROTEINS"
        super().__init__(dataset=TUDataset,config=config)

class BRZ_DATASET(BaseDataset):
    def __init__(self,config):
        #config.pre_transform = transforms.Compose([ThresholdTransform()])
        config.root = "./data"
        config.name="BRZ"
        super().__init__(dataset=TUDataset,config=config)


class LETTER_HIGH_DATASET(BaseDataset):
    def __init__(self,config):
        #config.pre_transform = transforms.Compose([ThresholdTransform()])
        config.root = "./data"
        config.name="Letter-high"
        super().__init__(dataset=TUDataset,config=config)



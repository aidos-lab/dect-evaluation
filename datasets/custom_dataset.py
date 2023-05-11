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

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

class TestMNIST(BaseDataset):
    """
    Example implementation for an existing dataset. 
    Note that we do the transform here, that is why we 
    need to create a separate class for the "new" 
    dataset.
    """
    def __init__(self,config):
        config.transform = transforms.Compose([transforms.ToTensor()])
        config.root = "./data"
        super().__init__(dataset=MNIST,config=config)
    
#
class LinearDataset(BaseDataset):
    """Represents a 2D segmentation dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(config=configuration)
        self.a = 2
        self.b = 3
        self.x = torch.linspace(0,1,100)
        self.y = self.a * self.x + self.b

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        # return the size of the dataset
        return self.x.shape[0]




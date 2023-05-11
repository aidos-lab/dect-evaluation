"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets. Also
    includes some transformation functions.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch.utils.data as data
import inspect
import torch
import torch_geometric
import torchvision.transforms as transforms


class BaseDataset(torch_geometric.data.Dataset):
    def __init__(self,dataset=None, config=None):
        super(BaseDataset,self).__init__()
        self.config = config
        if dataset:
            self.dataset = dataset(**vars(config))
        else:
            pass
        #print("Detected custom dataset, please implement __len__ and __getitem__")

    def len(self):
        #return self.dataset.len()
        return len(self.dataset)

    def get(self, idx):
        return self.dataset.__getitem__(idx)



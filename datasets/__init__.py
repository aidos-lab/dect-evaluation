import os, sys
import importlib
from pydoc import locate
import torch.utils.data as data
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

"""
This package includes all the modules related to data loading and preprocessing.
"""

"""
The below code fetches all files in the datasets folder and imports them. Each 
dataset can have its own file, or grouped in sets.
"""
path = os.path.dirname(os.path.abspath(__file__))
for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)


def get_datamodule(dataset_name):
    dataset = locate(f'datasets.{dataset_name}')
    if not dataset:
        print(dataset_name)
        raise AttributeError()
    return dataset

def load_datamodule(
        name=None,
        config=None
       ):
    dm = get_datamodule(name)
    return dm(config)




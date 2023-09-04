from torch_geometric.datasets import ModelNet
from torch_geometric import transforms
from datasets.base_dataset import DataModule, DataModuleConfig
from torch.utils.data import random_split
import torch
from loaders.factory import register
from dataclasses import dataclass
import numpy as np
import torch_geometric


@dataclass
class ModelNetDataModuleConfig(DataModuleConfig):
    root: str = "./data/modelnet"
    name: str = "10"
    module: str = "datasets.modelnet"
    samplepoints: int = 100


#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯


class ModelNetTransform(object):
    def __call__(self, data):
        data.x = data.pos
        data.pos = None
        return data


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


class Standardize(object):
    def __init__(self, samplepoints):
        self.samplepoints = samplepoints

    def __call__(self, data):
        x = data.pos
        data.pos = None
        clipper = torch.mean(torch.abs(x))
        z = torch.clip(x, -100 * clipper, 100 * clipper)
        mean = torch.mean(z)
        std = torch.std(z)
        normalized = (z - mean) / std
        data.x = normalized
        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯


class ModelNetPointsDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                transforms.SamplePoints(self.config.samplepoints),
                Standardize(self.config.samplepoints),
                CenterTransform(),
            ]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    def prepare_data(self):
        self.entire_ds = torch.utils.data.ConcatDataset(
            [
                ModelNet(
                    root=self.config.root,
                    pre_transform=self.pre_transform,
                    train=True,
                    name=self.config.name,
                ),
                ModelNet(
                    root=self.config.root,
                    pre_transform=self.pre_transform,
                    train=False,
                    name=self.config.name,
                ),
            ]
        )


def initialize():
    register("dataset", ModelNetPointsDataModule)

from torch_geometric.datasets import ModelNet
from torch_geometric import transforms
from datasets.base_dataset import DataModule, DataModuleConfig
from torch.utils.data import random_split
import torch
from loaders.factory import register
from dataclasses import dataclass
import numpy as np
from torch_geometric.transforms import FaceToEdge

from datasets.transforms import CenterTransform, SimplifyMesh


import itertools
from typing import Protocol
from dataclasses import dataclass
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold, train_test_split


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


class FloatTransform(object):
    def __call__(self, data):
        data.x = data.x.to(torch.float64)
        return data


class Rotate(object):
    def __call__(self, batch):
        theta = (torch.rand(1) - 0.5) * torch.pi / 5
        rot = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        batch.x = batch.x @ rot
        # scaling
        return batch


class Project(object):
    def __call__(self, batch):
        batch.x = batch.x[:, :2]
        # scaling
        return batch


#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯


class ModelNetMeshDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                SimplifyMesh(),
                FaceToEdge(remove_faces=False),
                CenterTransform(),
            ]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            # transform=FloatTransform(),
            train=True,
            name=self.config.name,
        )
        ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            # transform=FloatTransform(),
            train=False,
            name=self.config.name,
        )

    def setup(self):
        self.entire_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            # transform=FloatTransform(),
            train=True,
            name=self.config.name,
        )
        self.train_ds, self.val_ds = random_split(self.entire_ds, [int(0.8 * len(self.entire_ds)), len(self.entire_ds) - int(0.8 * len(self.entire_ds))])  # type: ignore
        self.test_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            # transform=FloatTransform(),
            train=False,
            name=self.config.name,
        )


def initialize():
    register("dataset", ModelNetMeshDataModule)

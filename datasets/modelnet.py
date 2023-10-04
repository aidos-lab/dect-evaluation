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


class ModelNetPointsDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                transforms.SamplePoints(self.config.samplepoints),
                ModelNetTransform(),
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
            # transform=Rotate(),
            train=True,
            name=self.config.name,
        )
        ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            # transform=Rotate(),
            train=False,
            name=self.config.name,
        )

    def setup(self):
        self.entire_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            # transform=Rotate(),
            train=True,
            name=self.config.name,
        )
        self.train_ds, self.val_ds = random_split(self.entire_ds, [int(0.8 * len(self.entire_ds)), len(self.entire_ds) - int(0.8 * len(self.entire_ds))])  # type: ignore
        self.test_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            # transform=Rotate(),
            train=False,
            name=self.config.name,
        )


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


# class ModelNetPointsDataModuleStratified(DataModule):
#     def __init__(self, config):
#         self.config = config
#         self.n_splits = 2
#         self.fold = 0
#         self.seed = 0
#         self.pre_transform = transforms.Compose(
#             [
#                 transforms.SamplePoints(self.config.samplepoints),
#                 # Standardize(self.config.samplepoints),
#                 CenterTransform(),
#             ]
#         )
#         super().__init__(
#             config.root, config.batch_size, config.num_workers, config.pin_memory
#         )

#     def prepare_data(self):
#         pass

#     def setup(self):
#         self.entire_ds = torch.utils.data.ConcatDataset(
#             [
#                 ModelNet(
#                     root=self.config.root,
#                     pre_transform=self.pre_transform,
#                     # transform=Rotate(),
#                     train=True,
#                     name=self.config.name,
#                 ),
#                 ModelNet(
#                     root=self.config.root,
#                     pre_transform=self.pre_transform,
#                     # transform=Rotate(),
#                     train=False,
#                     name=self.config.name,
#                 ),
#             ]
#         )
#         n_instances = len(self.entire_ds)
#         labels = torch.concat([data.y for data in self.entire_ds])
#         skf = StratifiedKFold(
#             n_splits=self.n_splits, random_state=self.seed, shuffle=True
#         )
#         skf_iterator = skf.split(
#             torch.tensor([i for i in range(n_instances)]),
#             torch.tensor(labels),
#         )
#         train_index, test_index = next(itertools.islice(skf_iterator, self.fold, None))
#         train_index, val_index = train_test_split(train_index, random_state=self.seed)
#         train_index = train_index.tolist()
#         val_index = val_index.tolist()
#         test_index = test_index.tolist()
#         self.train_ds = Subset(self.entire_ds, train_index)
#         self.val_ds = Subset(self.entire_ds, val_index)
#         self.test_ds = Subset(self.entire_ds, test_index)


def initialize():
    register("dataset", ModelNetPointsDataModule)

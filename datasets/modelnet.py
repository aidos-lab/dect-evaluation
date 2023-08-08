from torch_geometric.datasets import ModelNet
from torch_geometric import transforms
from datasets.base_dataset import DataModule, DataModuleConfig
from torch.utils.data import random_split
import torch
from loaders.factory import register
from dataclasses import dataclass


@dataclass
class ModelNetDataModuleConfig(DataModuleConfig):
    root: str = "./data/modelnet100"
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

    def prepare_data(self):
        ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=True,
            name=self.config.name,
        )
        ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=False,
            name=self.config.name,
        )

    def setup(self):
        self.entire_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=True,
            name=self.config.name,
        )
        self.train_ds, self.val_ds = random_split(self.entire_ds, [0.8, 0.2])  # type: ignore
        self.test_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=False,
            name=self.config.name,
        )

    """ def info(self): """
    """     print("len train_ds", len(self.train_ds)) """
    """     print("len val_ds", len(self.val_ds)) """
    """     print("len test_ds", len(self.test_ds)) """
    """     print(self.train_ds) """
    """     print(self.val_ds) """
    """     print("Bincount test",torch.bincount(self.test_ds.y, minlength=10)) """
    """     print("Bincount train",torch.bincount(self.entire_ds.y, minlength=10)) """
    """"""
    """     counts = torch.zeros(10) """
    """     for data in self.train_dataloader(): """
    """         counts += torch.bincount(data.y,minlength=10) """
    """     print("Bincount train",counts) """
    """     counts = torch.zeros(10) """
    """     for data in self.val_dataloader(): """
    """         counts += torch.bincount(data.y,minlength=10) """
    """     print("Bincount val",counts) """
    """     counts = torch.zeros(10) """
    """     for data in self.test_dataloader(): """
    """         counts += torch.bincount(data.y,minlength=10) """
    """     print("Bincount test",counts) """


class ModelNetMeshDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )
        self.pre_transform = transforms.Compose(
            [ModelNetTransform(), CenterTransform()]
        )
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=True,
            name=self.config.name,
        )
        ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=False,
            name=self.config.name,
        )

    def setup(self):
        entire_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=True,
            name=self.config.name,
        )
        self.train_ds, self.val_ds = random_split(entire_ds, [int(0.8 * len(entire_ds)), len(entire_ds) - int(0.8 * len(entire_ds))])  # type: ignore
        self.test_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=False,
            name=self.config.name,
        )


def initialize():
    register("dataset", ModelNetPointsDataModule)

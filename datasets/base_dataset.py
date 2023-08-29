from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader, ImbalancedSampler
from torch_geometric.data import Dataset
import torch

from typing import Protocol
from dataclasses import dataclass


@dataclass
class DataModuleConfig(Protocol):
    module: str
    root: str = "./data"
    num_workers: int = 0
    batch_size: int = 64
    pin_memory: bool = True
    drop_last: bool = False


class DataModule(ABC):
    train_ds: Dataset
    test_ds: Dataset
    val_ds: Dataset
    entire_ds: Dataset

    def __init__(self, root, batch_size, num_workers, pin_memory=True, drop_last=True):
        super().__init__()
        self.data_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prepare_data()
        self.setup()
        # self.info()

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError()

    @abstractmethod
    def setup(self):
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # sampler=ImbalancedSampler(self.train_ds),
            shuffle=True,
            pin_memory=self.pin_memory,
            # drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # sampler=ImbalancedSampler(self.val_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
            # drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # sampler=ImbalancedSampler(self.test_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
            # drop_last=self.drop_last,
        )

    def info(self):
        print("The train dataset contains ", len(self.train_ds), "elements.")
        print("The validation dataset contains ", len(self.val_ds), "elements.")
        print("The test dataset contains ", len(self.test_ds), "elements.")
        print("The number of classes", self.entire_ds.num_classes)
        print("The shape of the node features is", self.train_ds.shape)
        print("An element of the train dataset:", self.train_ds[0])

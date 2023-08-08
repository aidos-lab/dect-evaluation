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



class DataModule(ABC):
    train_ds: Dataset
    test_ds: Dataset
    val_ds: Dataset
    entire_ds: Dataset

    def __init__(self, root, batch_size, num_workers, pin_memory=True):
        super().__init__()
        self.data_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prepare_data()
        self.setup()

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
            sampler=ImbalancedSampler(self.train_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=ImbalancedSampler(self.val_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=ImbalancedSampler(self.test_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def info(self):
        print("len train_ds", len(self.train_ds))
        print("len val_ds", len(self.val_ds))
        print("len test_ds", len(self.test_ds))
        print("data num_classes", self.entire_ds.num_classes)
        print(self.train_ds)
        print(self.val_ds)
        print(self.train_ds[0])
        counts = torch.zeros(self.entire_ds.num_classes)
        for data in self.train_dataloader():
            counts += torch.bincount(data.y, minlength=10)
        print("Bincount train", counts)
        counts = torch.zeros(self.entire_ds.num_classes)
        for data in self.val_dataloader():
            counts += torch.bincount(data.y, minlength=10)
        print("Bincount val", counts)
        counts = torch.zeros(self.entire_ds.num_classes)
        for data in self.test_dataloader():
            counts += torch.bincount(data.y, minlength=10)
        print("Bincount test", counts)

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

<<<<<<< HEAD
    def __init__(self, root, batch_size, num_workers, pin_memory=True, drop_last=True):
=======
    def __init__(self, root, batch_size, num_workers, pin_memory=True):
>>>>>>> main
        super().__init__()
        self.data_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
<<<<<<< HEAD
        self.drop_last = drop_last
        self.prepare_data()
        self.setup()
        # self.info()
=======
        self.prepare_data()
        self.setup()
>>>>>>> main

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
<<<<<<< HEAD
            # sampler=ImbalancedSampler(self.train_ds),
            shuffle=True,
            pin_memory=self.pin_memory,
            # drop_last=self.drop_last,
=======
            sampler=ImbalancedSampler(self.train_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
>>>>>>> main
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
<<<<<<< HEAD
            # sampler=ImbalancedSampler(self.val_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
            # drop_last=self.drop_last,
=======
            sampler=ImbalancedSampler(self.val_ds),
            shuffle=False,
            pin_memory=self.pin_memory,
>>>>>>> main
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
<<<<<<< HEAD
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
=======
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
>>>>>>> main

from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader, ImbalancedSampler
from torch_geometric.data import Dataset
import torch
import torch_geometric
import itertools
from typing import Protocol
from dataclasses import dataclass
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class DataModuleConfig(Protocol):
    module: str
    root: str = "./data"
    num_workers: int = 0
    batch_size: int = 64
    pin_memory: bool = True
    drop_last: bool = False


class DataModule(ABC):
    train_ds: Dataset | None
    test_ds: Dataset | None
    val_ds: Dataset | None
    entire_ds: Dataset | None

    def __init__(self, root, batch_size, num_workers, pin_memory=True, drop_last=True):
        super().__init__()
        self.seed = 4338
        self.n_splits = 3
        self.fold = 0
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

    def setup(self):
        # # Prevents previous experiments to be overwritten.
        # if self.train_ds and self.test_ds and self.val_ds:
        #     return

        n_instances = len(self.entire_ds)
        labels = torch.concat([data.y for data in self.entire_ds])
        print(labels.shape)
        skf = StratifiedKFold(
            n_splits=self.n_splits, random_state=self.seed, shuffle=True
        )
        skf_iterator = skf.split(
            torch.tensor([i for i in range(n_instances)]),
            torch.tensor(labels),
        )
        train_index, test_index = next(itertools.islice(skf_iterator, self.fold, None))
        train_index, val_index = train_test_split(train_index, random_state=self.seed)
        train_index = train_index.tolist()
        val_index = val_index.tolist()
        test_index = test_index.tolist()
        self.train_ds = Subset(self.entire_ds, train_index)
        self.val_ds = Subset(self.entire_ds, val_index)
        self.test_ds = Subset(self.entire_ds, test_index)

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

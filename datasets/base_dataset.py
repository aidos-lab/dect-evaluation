from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


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

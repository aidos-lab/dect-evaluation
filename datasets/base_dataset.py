from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader

class DataModule():
    def __init__(self, root, batch_size, num_workers):
        super().__init__()
        self.data_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod    
    def setup(self, stage):
        pass


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )




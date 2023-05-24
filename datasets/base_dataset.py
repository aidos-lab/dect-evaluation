from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


class DataModule(ABC):
    train_ds: Dataset
    test_ds: Dataset
    val_ds: Dataset

    def __init__(self, root, batch_size, num_workers):
        super().__init__()
        self.data_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __post_init__(self): 
        self.prepare_data()
        self.setup()
    

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod    
    def setup(self):
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




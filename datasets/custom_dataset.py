from datasets.base_dataset import DataModule
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
<<<<<<< HEAD
=======

#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯
>>>>>>> main

from dataclasses import dataclass
from loaders.factory import register
from datasets.base_dataset import DataModuleConfig


@dataclass
class MnistConfig(DataModuleConfig):
    module: str = "datasets.custom_dataset"
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory = False



class MnistDataModule(DataModule):
    """
    Example implementation for an existing dataset.
    Note that we do the transform here, that is why we
    need to create a separate class for the "new"
    dataset.
    """

<<<<<<< HEAD
    def __init__(self, config: MnistConfig):
        super().__init__(config)
        self.config = config

    def setup(self):
        """
        This method constructs the entire dataset, note we concatenate the train/test datasets.
        This allows for k-fold cross validation later on.
        If you prefer to create the splits yourself, set the variables
        self.train_ds, self.test_ds and self.val_ds.
        """
        entire_dataset = torch.utils.data.ConcatDataset(
            [
                MNIST(
                    root=self.config.data_dir,
                    train=True,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                ),
                MNIST(
                    root=self.config.data_dir,
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                ),
            ]
=======
    def __init__(self, config):
        super().__init__(config.data_dir, config.batch_size, config.num_workers)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
>>>>>>> main
        )
        return entire_dataset


<<<<<<< HEAD
def initialize():
    register("dataset", MnistDataModule)
=======
class LinearDataModule(DataModule):
    def __init__(self, config):
        super().__init__(config.data_dir, config.batch_size, config.num_workers)

    def prepare_data(self):
        pass

    def setup(self, stage):
        entire_dataset = LinearDataset()
        self.train_ds, self.val_ds = random_split(entire_dataset, [80, 20])
        self.test_ds = entire_dataset


class LinearDataset:
    """Represents a 2D segmentation dataset.

    Input params:
        configuration: Configuration dictionary.
    """

    def __init__(self):
        self.a = 2
        self.b = 3
        self.x = torch.linspace(0, 1, 100)
        self.y = self.a * self.x + self.b

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        # return the size of the dataset
        return self.x.shape[0]
>>>>>>> main

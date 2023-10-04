import torch
from torch.utils.data import random_split

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import FaceToEdge

from datasets.transforms import CenterTransform
from datasets.base_dataset import DataModule
from datasets.transforms import MnistTransform

from loaders.factory import register


class MnistDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [MnistTransform(), FaceToEdge(remove_faces=False), CenterTransform()]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    def setup(self):
        self.entire_ds = MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=True
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.9 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.9 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = MnistDataset(
            root=self.config.root, pre_transform=self.transform, train=False
        )


class MnistDataset(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, train=True, pre_filter=None
    ):
        self.train = train
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ["MNIST"]

    @property
    def processed_file_names(self):
        return ["train.pt", "test.pt"]

    def download(self):
        if self.train:
            MNIST(f"{self.root}/raw/", train=True, download=True)
        else:
            MNIST(f"{self.root}/raw/", train=False, download=True)

    def process(self):
        train_ds = MNIST(f"{self.root}/raw/", train=True, download=True)
        test_ds = MNIST(f"{self.root}/raw/", train=False, download=True)

        if self.pre_transform is not None:
            train_data_list = [self.pre_transform(data) for data in train_ds]
            test_data_list = [self.pre_transform(data) for data in test_ds]

        train_data, train_slices = self.collate(train_data_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
        test_data, test_slices = self.collate(test_data_list)
        torch.save((test_data, test_slices), self.processed_paths[1])


def initialize():
    register("dataset", MnistDataModule)

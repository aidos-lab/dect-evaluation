import vedo
import torch
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
import torchvision
from torch_geometric.transforms import FaceToEdge
import numpy as np
from datasets.base_dataset import DataModule, DataModuleConfig
from dataclasses import dataclass
from torch_geometric.data import Dataset, Data
import torch.functional as F
from torch.utils.data import random_split
from datasets.transforms import CenterTransform
import torch
from torch_geometric.data import InMemoryDataset, download_url
from loaders.factory import register


class FashionMnistTransform:
    def __init__(self):
        xcoords = torch.linspace(-0.5, 0.5, 28)
        ycoords = torch.linspace(-0.5, 0.5, 28)
        self.X, self.Y = torch.meshgrid(xcoords, ycoords)
        self.tr = torchvision.transforms.ToTensor()

    def __call__(self, data: tuple) -> Data:
        img, y = data
        img = self.tr(img)
        idx = torch.nonzero(img.squeeze(), as_tuple=True)
        gp = torch.vstack([self.X[idx], self.Y[idx]]).T
        dly = vedo.delaunay2d(gp, mode="xy", alpha=0.03).c("w").lc("o").lw(1)
        # print(torch.tensor(dly.edges()).T.shape)
        # print(torch.tensor(dly.faces()).T.shape)
        # print(torch.tensor(dly.points()).shape)

        return Data(
            x=torch.tensor(dly.points()),
            face=torch.tensor(dly.faces(), dtype=torch.long).T,
            y=torch.tensor(y, dtype=torch.long),
        )


@dataclass
class FashionMnistDataModuleConfig(DataModuleConfig):
    root: str = "./data/FashionMNIST"
    module: str = "datasets.fashion_mnist"


class FashionMnistDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [FashionMnistTransform(), FaceToEdge(remove_faces=False), CenterTransform()]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    def prepare_data(self):
        self.entire_ds = FashionMnistDataset(
            root=self.config.root, pre_transform=self.transform, train=True
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.9 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.9 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = FashionMnistDataset(
            root=self.config.root, pre_transform=self.transform, train=False
        )

    def setup(self):
        pass


class FashionMnistDataset(InMemoryDataset):
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
        return ["FashionMNIST"]

    @property
    def processed_file_names(self):
        return ["train.pt", "test.pt"]

    def download(self):
        if self.train:
            FashionMNIST(f"{self.root}/raw/", train=True, download=True)
        else:
            FashionMNIST(f"{self.root}/raw/", train=False, download=True)

    def process(self):
        train_ds = FashionMNIST(f"{self.root}/raw/", train=True, download=True)
        test_ds = FashionMNIST(f"{self.root}/raw/", train=False, download=True)

        if self.pre_transform is not None:
            train_data_list = [self.pre_transform(data) for data in train_ds]
            test_data_list = [self.pre_transform(data) for data in test_ds]

        train_data, train_slices = self.collate(train_data_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
        test_data, test_slices = self.collate(test_data_list)
        torch.save((test_data, test_slices), self.processed_paths[1])


def initialize():
    register("dataset", FashionMnistDataModule)

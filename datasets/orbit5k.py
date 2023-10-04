import torch
import torchvision.transforms as transforms
import numpy as np
from datasets.base_dataset import DataModule, DataModuleConfig
from dataclasses import dataclass
from torch_geometric.data import Data
from torch.utils.data import random_split
from datasets.transforms import CenterTransform
import torch
from torch_geometric.transforms import FaceToEdge

from torch_geometric.data import InMemoryDataset
from loaders.factory import register
import torchvision
import vedo


class Orbit5kTransform:
    def __call__(self, data: tuple) -> Data:
        X, y = data
        dly = (
            vedo.delaunay2d(torch.tensor(X, dtype=torch.float), mode="xy", alpha=0.07)
            .c("w")
            .lc("o")
            .lw(1)
        )
        return Data(
            x=torch.tensor(dly.points()),
            face=torch.tensor(dly.faces(), dtype=torch.long).T,
            y=torch.tensor([y], dtype=torch.long),
        )


@dataclass
class Orbit5kDataModuleConfig(DataModuleConfig):
    root: str = "./data/orbit5k"
    module: str = "datasets.orbit5k"


class Orbit5kDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [Orbit5kTransform(), FaceToEdge(remove_faces=False), CenterTransform()]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    def prepare_data(self):
        self.entire_ds = Orbit5kDataset(
            root=self.config.root, pre_transform=self.transform, train=True
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.2 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.2 * len(self.entire_ds)),
            ],
        )  # type: ignore

        self.test_ds = Orbit5kDataset(
            root=self.config.root, pre_transform=self.transform, train=False
        )

    def setup(self):
        pass


class Orbit5kDataset(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, train=True, pre_filter=None
    ):
        self.train = train
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.path = self.processed_paths[0] if self.train else self.processed_paths[1]
        self.data, self.slices = torch.load(self.path)

    @property
    def raw_file_names(self):
        return [
            "orbit5k_X_original_00_train.npy",
            "orbit5k_y_train.npy",
            "orbit5k_X_original_00_test.npy",
            "orbit5k_y_test.npy",
        ]

    @property
    def processed_file_names(self):
        return ["train.pt", "test.pt"]

    def download(self):
        pass

    def process(self):
        data_X_train = np.load(f"{self.root}/raw/" + self.raw_file_names[0])
        data_y_train = np.load(f"{self.root}/raw/" + self.raw_file_names[1])
        data_X_test = np.load(f"{self.root}/raw/" + self.raw_file_names[2])
        data_y_test = np.load(f"{self.root}/raw/" + self.raw_file_names[3])

        if self.pre_transform is not None:
            train_data_list = [
                self.pre_transform(data) for data in zip(data_X_train, data_y_train)
            ]
            test_data_list = [
                self.pre_transform(data) for data in zip(data_X_test, data_y_test)
            ]

        train_data, train_slices = self.collate(train_data_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
        test_data, test_slices = self.collate(test_data_list)
        torch.save((test_data, test_slices), self.processed_paths[1])


def initialize():
    register("dataset", Orbit5kDataModule)


if __name__ == "__main__":
    dataset = Orbit5kDataModule(Orbit5kDataModuleConfig())
    print(dataset.train_ds[0])

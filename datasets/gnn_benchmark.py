from datasets.base_dataset import DataModule
from torch_geometric.datasets import GNNBenchmarkDataset

import torchvision.transforms as transforms
from datasets.transforms import CenterTransform, ThresholdTransform

from loaders.factory import register


transforms_dict = {
    "MNIST": [
        ThresholdTransform(),
        CenterTransform(),
    ],
    "CIFAR10": [
        ThresholdTransform(),
        CenterTransform(),
    ],
    "PATTERN": [
        CenterTransform(),
    ],
}


class GNNBenchmarkDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(transforms_dict[self.config.name])
        super().__init__(config.root, config.batch_size, config.num_workers)

    def setup(self):
        self.train_ds = GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="train",
        )
        self.test_ds = GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="test",
        )
        self.val_ds = GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="val",
        )


def initialize():
    register("dataset", GNNBenchmarkDataModule)

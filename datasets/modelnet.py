from torch_geometric.datasets import ModelNet
from torch_geometric import transforms
from datasets.base_dataset import DataModule
from torch.utils.data import random_split
import torch

from torch_geometric.transforms import FaceToEdge
from datasets.transforms import CenterTransform, SimplifyMesh, ModelNetTransform

from loaders.factory import register


class ModelNetDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.pre_transform = transforms.Compose(
            [
                transforms.SamplePoints(self.config.samplepoints),
                ModelNetTransform(),
                CenterTransform(),
            ]
        )
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

        self.prepare_data()
        self.setup()

    def setup(self):
        self.entire_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=True,
            name=self.config.name,
        )
        self.train_ds, self.val_ds = random_split(
            self.entire_ds,
            [
                int(0.8 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.8 * len(self.entire_ds)),
            ],
        )  # type: ignore
        self.test_ds = ModelNet(
            root=self.config.root,
            pre_transform=self.pre_transform,
            train=False,
            name=self.config.name,
        )


def initialize():
    register("dataset", ModelNetDataModule)

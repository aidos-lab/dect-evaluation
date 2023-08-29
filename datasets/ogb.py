from datasets.base_dataset import DataModule, DataModuleConfig

# from base_dataset import DataModule
from torch_geometric.datasets import TUDataset
from dataclasses import dataclass
from torch.utils.data import random_split
from torch_geometric import transforms
from loaders.factory import register
from ogb.graphproppred import PygGraphPropPredDataset

#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯

"""
Add all the required transforms in this section, or use imports.
"""


class CenterTransform(object):
    def __call__(self, data):
        data.x = data.x.float()
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

"""
Define the dataset classes, provide dataset/dataloader parameters 
in the config file or overwrite them in the class definition.
"""


@dataclass
class MOLHIVDataModuleConfig(DataModuleConfig):
    name: str = "ogbg-molhiv"
    module: str = "datasets.ogb"


class OGBDataModule(DataModule):
    """
    Write stuff here ;)
    """

    def __init__(self, config):
        self.config = config
        super().__init__(config.root, config.batch_size, config.num_workers)

    def prepare_data(self):
        PygGraphPropPredDataset(
            pre_transform=transforms.Compose([CenterTransform()]),
            name=self.config.name,
            root=self.config.root,
        )

    def setup(self):
        entire_ds = PygGraphPropPredDataset(
            pre_transform=transforms.Compose([CenterTransform()]),
            name=self.config.name,
            root=self.config.root,
        )
        inter_ds, self.test_ds = random_split(
            entire_ds,
            [int(0.9 * len(entire_ds)), len(entire_ds) - int(0.9 * len(entire_ds))],
        )  # type: ignore
        self.train_ds, self.val_ds = random_split(inter_ds, [int(0.9 * len(inter_ds)), len(inter_ds) - int(0.9 * len(inter_ds))])  # type: ignore


def initialize():
    register("dataset", OGBDataModule)

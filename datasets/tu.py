import torch

from datasets.base_dataset import DataModule, DataModuleConfig

# from base_dataset import DataModule
from torch_geometric.datasets import TUDataset
from dataclasses import dataclass
from torch.utils.data import random_split
from torch_geometric import transforms
from loaders.factory import register
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree


#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯

"""
Add all the required transforms in this section, or use imports.
"""


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


class Normalize(object):
    def __call__(self, data):
        mean = data.x.mean()
        std = data.x.std()
        data.x = (data.x - mean) / std
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


transforms_dict = {
    "DD": [CenterTransform()],
    "ENZYMES": [CenterTransform()],
    "IMDB-BINARY": [OneHotDegree(540), CenterTransform()],
    "Letter-high": [CenterTransform()],
    "Letter-med": [CenterTransform()],
    "Letter-low": [CenterTransform()],
    "PROTEINS_full": [CenterTransform()],
    "REDDIT-BINARY": [
        NormalizedDegree(2.31, 20.66),
        CenterTransform(),
    ],
    "NCI1": [CenterTransform()],
    "NCI109": [],
}

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯


"""
Define the dataset classes, provide dataset/dataloader parameters 
in the config file or overwrite them in the class definition.
"""


@dataclass
class TUBBBPConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "BBBP"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUNCI109Config(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "NCI109"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUNCI1Config(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "NCI1"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUDDConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "DD"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUEnzymesConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "ENZYMES"
    cleaned: bool = False
    use_node_attr: bool = True


@dataclass
class TUIMDBBConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "IMDB-BINARY"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUProteinsFullConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "PROTEINS_full"
    cleaned: bool = False
    use_node_attr: bool = True


@dataclass
class TURedditBConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "REDDIT-BINARY"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TULetterHighConfig(DataModuleConfig):
    name: str = "Letter-high"
    module: str = "datasets.tu"
    cleaned: bool = False
    drop_last: bool = False


@dataclass
class TULetterMedConfig(DataModuleConfig):
    name: str = "Letter-med"
    module: str = "datasets.tu"
    cleaned: bool = False
    drop_last: bool = False


@dataclass
class TULetterLowConfig(DataModuleConfig):
    name: str = "Letter-low"
    module: str = "datasets.tu"
    cleaned: bool = False
    drop_last: bool = False


class TUDataModule(DataModule):
    """
    This datamodule loads the base TUDatasets without transforming.
    See below how to add a transform the easiest way.
    """

    def __init__(self, config):
        self.config = config
        super().__init__(
            config.root,
            config.batch_size,
            config.num_workers,
            drop_last=self.config.drop_last,
        )

    def prepare_data(self):
        pass

    def setup(self):
        self.entire_ds = TUDataset(
            pre_transform=transforms.Compose(transforms_dict[self.config.name]),
            name=self.config.name,
            root=self.config.root,
            cleaned=self.config.cleaned,
            use_node_attr=True,
        )
        inter_ds, self.test_ds = random_split(
            self.entire_ds,
            [
                int(0.8 * len(self.entire_ds)),
                len(self.entire_ds) - int(0.8 * len(self.entire_ds)),
            ],
        )  # type: ignore
        self.train_ds, self.val_ds = random_split(inter_ds, [int(0.8 * len(inter_ds)), len(inter_ds) - int(0.8 * len(inter_ds))])  # type: ignore


def initialize():
    register("dataset", TUDataModule)

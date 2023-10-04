from torch_geometric.datasets import TUDataset

from torch.utils.data import random_split

from torch_geometric import transforms
from torch_geometric.transforms import OneHotDegree

from datasets.config import DataModuleConfig
from datasets.base_dataset import DataModule
from datasets.transforms import (
    CenterTransform,
    NormalizedDegree,
    NCI109Transform,
)

from dataclasses import dataclass
from loaders.factory import register


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
    "NCI1": [NCI109Transform(), CenterTransform()],
    "NCI109": [CenterTransform()],
    "BZR": [CenterTransform()],
    "COX2": [CenterTransform()],
    "FRANKENSTEIN": [CenterTransform()],
    "Fingerprint": [CenterTransform()],
    "Cuneiform": [CenterTransform()],
    "COLLAB": [CenterTransform()],
    "DHFR": [CenterTransform()],
}

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯


"""
Define the dataset classes, provide dataset/dataloader parameters 
in the config file or overwrite them in the class definition.
"""


@dataclass
class TUBZRConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "BZR"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUCOX2Config(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "COX2"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUFrankensteinConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "FRANKENSTEIN"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUFingerprintConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "Fingerprint"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUCuneiformConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "Cuneiform"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUCollabConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "COLLAB"
    cleaned: bool = True
    use_node_attr: bool = True


@dataclass
class TUDHFRConfig(DataModuleConfig):
    module: str = "datasets.tu"
    name: str = "DHFR"
    cleaned: bool = True
    use_node_attr: bool = True


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

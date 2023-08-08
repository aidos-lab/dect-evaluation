from datasets.base_dataset import DataModule, DataModuleConfig

# from base_dataset import DataModule
from torch_geometric.datasets import TUDataset
from dataclasses import dataclass
from torch.utils.data import random_split
from torch_geometric import transforms
from loaders.factory import register

#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯

"""
Add all the required transforms in this section, or use imports.
"""

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

"""
Define the dataset classes, provide dataset/dataloader parameters 
in the config file or overwrite them in the class definition.
"""


@dataclass
class TUDataModuleConfig(DataModuleConfig):
    name: str = "Letter-high"
    module: str = "datasets.tu"


class TUDataModule(DataModule):
    """
    This datamodule loads the base TUDatasets without transforming.
    See below how to add a transform the easiest way.
    """

    def __init__(self, config: TUDataModuleConfig):
        self.config = config
        super().__init__(config.root, config.batch_size, config.num_workers)

    def prepare_data(self):
        TUDataset(name=self.config.name, root=self.config.root, use_node_attr=True)

    def setup(self):
        print("le")
        entire_ds = TUDataset(
            name=self.config.name, root=self.config.root, use_node_attr=True
        )
        inter_ds, self.test_ds = random_split(entire_ds, [int(0.8 * len(entire_ds)), len(entire_ds) - int(0.8 * len(entire_ds))])  # type: ignore
        self.train_ds, self.val_ds = random_split(inter_ds, [int(0.8 * len(inter_ds)), len(inter_ds) - int(0.8 * len(inter_ds))])  # type: ignore


class TULetterHighDataModule(DataModule):
    """
    This datamodule loads the base TUDatasets without transforming.
    See below how to add a transform the easiest way.
    """

    def __init__(self, config):
        self.config = config
        super().__init__(config.root, config.batch_size, config.num_workers)

    def prepare_data(self):
        TUDataset(
            pre_transform=transforms.Compose([]),
            name=self.config.name,
            root=self.config.root,
            use_node_attr=True,
        )

    def setup(self):
        entire_ds = TUDataset(
            pre_transform=transforms.Compose([]),
            name=self.config.name,
            root=self.config.root,
            use_node_attr=True,
        )
        inter_ds, self.test_ds = random_split(entire_ds, [int(0.8 * len(entire_ds)), len(entire_ds) - int(0.8 * len(entire_ds))])  # type: ignore
        self.train_ds, self.val_ds = random_split(inter_ds, [int(0.8 * len(inter_ds)), len(inter_ds) - int(0.8 * len(inter_ds))])  # type: ignore


def initialize():
    register("dataset", TULetterHighDataModule)

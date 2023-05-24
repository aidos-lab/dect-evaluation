from torch_geometric.datasets import ModelNet
from torch_geometric import transforms
from datasets.base_dataset import DataModule
from torch.utils.data import random_split

#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯

class ModelNetTransform(object):
  def __call__(self, data):
    data.x = data.pos
    return data

class CenterTransform(object):
  def __call__(self, data):
    data.x -= data.x.mean()
    data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
    return data

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

class ModelNetPointsDataModule(DataModule):
    def __init__(self,config):
        super().__init__(config.root,config.batch_size,config.num_workers)
        self.config = config
        self.pre_transform = transforms.Compose([transforms.SamplePoints(self.config.samplepoints),
                                                   ModelNetTransform(),
                                                   CenterTransform()])
    
    def prepare_data(self):
        ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = True
                )
        ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = False
                )

    def setup(self):
        entire_ds = ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = True
                )
        self.train_ds, self.val_ds = random_split(entire_ds, [int(0.8*len(entire_ds)), len(entire_ds)-int(0.8*len(entire_ds))]) # type: ignore
        self.test_ds = ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = False
                )

class ModelNetMeshDataModule(DataModule):
    def __init__(self,config):
        super().__init__(config.root,config.batch_size,config.num_workers)
        self.config = config
        self.pre_transform = transforms.Compose([ModelNetTransform(),
                                                   CenterTransform()])
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = True
                )
        ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = False
                )

    def setup(self):
        entire_ds = ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = True
                )
        self.train_ds, self.val_ds = random_split(entire_ds, [int(0.8*len(entire_ds)), len(entire_ds)-int(0.8*len(entire_ds))]) # type: ignore
        self.test_ds = ModelNet(
                root = self.config.root,
                pre_transform=self.pre_transform,
                train = False
                )


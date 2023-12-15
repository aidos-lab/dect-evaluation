from dataclasses import dataclass


@dataclass
class DataModuleConfig:
    module: str
    root: str = "./data"
    num_workers: int = 0
    batch_size: int = 64
    pin_memory: bool = True
    drop_last: bool = False


@dataclass
class GNNBenchmarkDataModuleConfig(DataModuleConfig):
    name: str = "MNIST"
    module: str = "datasets.gnn_benchmark"


@dataclass
class MnistDataModuleConfig(DataModuleConfig):
    root: str = "./data/MNIST"
    module: str = "datasets.mnist"

    
@dataclass
class WeightedMnistDataModuleConfig(DataModuleConfig):
    root: str = "./data/WMNIST"
    module: str = "datasets.weighted_mnist"


@dataclass
class ModelNetDataModuleConfig(DataModuleConfig):
    root: str = "./data/modelnet"
    name: str = "10"
    module: str = "datasets.modelnet"
    samplepoints: int = 100

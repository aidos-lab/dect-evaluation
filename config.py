from typing import Any, Protocol
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    meta: Any
    data: Any
    model: Any
    trainer: Any


<<<<<<< HEAD
@dataclass
class Meta:
    name: str
    project: str = "desct"
    tags: list[str] = field(default_factory=list)
    experiment_folder: str = "experiment"


=======
>>>>>>> main
#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯

<<<<<<< HEAD
=======

@dataclass
class DataModuleConfig:
    name: str
    config: Any


@dataclass
class GNNBenchmarkConfig:
    name: str = "MNIST"
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 0
    split: str = "train"
    pin_memory: bool = True


@dataclass
class TUDataConfig:
    name: str = "Letter-high"
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class ModelnetConfig:
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 0
    samplepoints: int = 100
    pin_memory: bool = True
    name: str = "10"


@dataclass
class ManifoldConfig:
    root: str = "./data/Manifold"
    batch_size: int = 128
    num_workers: int = 0
    samplepoints: int = 100
    num_samples: int = 25
    pin_memory: bool = True

>>>>>>> main

#  ╭──────────────────────────────────────────────────────────╮
#  │ Model Configurations                                     │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class ModelConfig:
    name: str
    config: Any
<<<<<<< HEAD

=======


@dataclass
class ECTLinearModelConfig:
    num_thetas: int
    hidden: int
    bump_steps: int
    batch_size: int = 128
    R: float = 1.2
    scale: int = 500
    num_features: int = 3
    num_classes: int = 100


@dataclass
class ECTCNNModelConfig:
    num_thetas: int
    bump_steps: int
    batch_size: int = 128
    R: float = 1.1
    scale: int = 500
    num_features: int = 3
    num_classes: int = 10
    hidden: int = 100
>>>>>>> main


#  ╭──────────────────────────────────────────────────────────╮
#  │ Trainer configurations                                   │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class TrainerConfig:
    lr: float = 0.001
    num_epochs: int = 200
<<<<<<< HEAD
    num_reruns: int = 1
=======
>>>>>>> main

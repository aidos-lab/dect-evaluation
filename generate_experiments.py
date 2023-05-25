import os 
from omegaconf import OmegaConf
import itertools
import shutil

from typing import Any
from dataclasses import dataclass

"""
This script creates all the configurations in the config folder. 
It allows for better reproducebility. The script takes 
NOTE: Gets ran every time make run is called.
"""


@dataclass(frozen=True)
class Config:
    data: Any
    model: Any
    trainer: Any

#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯

@dataclass
class DataModuleConfig:
  name: str
  config: Any

@dataclass
class GNNBenchmarkConfig:
    name: str = "MNIST"
    root: str = "./data"
    batch_size: int = 256
    num_workers: int = 16
    split: str = "train"

@dataclass
class ModelnetConfig:
    root: str = "./data/Manifold"
    batch_size: int = 64
    num_workers: int = 16
    samplepoints: int = 100

@dataclass
class ManifoldConfig:
    root: str = "./data"
    batch_size: int = 256
    num_workers: int = 16
    samplepoints: int = 100
    num_samples: int = 25


#  ╭──────────────────────────────────────────────────────────╮
#  │ Model Configurations                                     │
#  ╰──────────────────────────────────────────────────────────╯

@dataclass
class ModelConfig:
  name: str
  config: Any

@dataclass
class ECTLinearModelConfig:
    num_thetas : int
    hidden: int
    bump_steps : int 
    R : float = 1.5
    scale : int = 500
    num_features : int = 3
    num_classes: int = 10

@dataclass
class ECTCNNModelConfig:
    num_thetas : int
    bump_steps : int 
    R : float = 1.5
    scale : int = 500
    num_features : int = 3
    num_classes: int = 10

#  ╭──────────────────────────────────────────────────────────╮
#  │ Trainer configurations                                   │
#  ╰──────────────────────────────────────────────────────────╯

@dataclass
class TrainerConfig:
    lr: float = 0.001
    num_epochs: int = 200

#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper methods                                           │
#  ╰──────────────────────────────────────────────────────────╯

def create_experiment_folder(path):
    shutil.rmtree(path,ignore_errors=True)
    os.makedirs(path)

def save_config(cfg,path):
    c = OmegaConf.create(cfg)
    with open(path,"w") as f:
        OmegaConf.save(c,f)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Experiments                                              │
#  ╰──────────────────────────────────────────────────────────╯

def theta_sweep():
    """
    This experiment trains two models with varying number of angles used. 
    The configs are stored in separate folders so the two experiments
    can be ran independently (it takes a while).
    """

    linear_experiment = "./experiment/linear_theta_sweep"
    create_experiment_folder(linear_experiment)
    
    cnn_experiment = "./experiment/cnn_theta_sweep"
    create_experiment_folder(cnn_experiment)

    # Create data config
    data = DataModuleConfig(
            name="GNNBenchmarkDataModule",
            config=GNNBenchmarkConfig())

    # Create Trainer Config
    trainer = TrainerConfig()

    for idx, num_thetas in enumerate(range(5,55,5)): 
        linear_model = ModelConfig(
                name="ECTLinearModel",
                config = ECTLinearModelConfig(bump_steps=20, hidden=10, num_thetas=num_thetas))

        cnn_model = ModelConfig(
                name="ECTCNNModel",
                config = ECTCNNModelConfig(bump_steps=20, num_thetas=num_thetas))

        linear_config = Config(data,linear_model,trainer)
        cnn_config = Config(data,cnn_model,trainer)

        save_config(linear_config,os.path.join(linear_experiment,f"{idx}.yaml"))
        save_config(cnn_config,os.path.join(cnn_experiment,f"{idx}.yaml"))


def modelnet_classification():
    """
    This experiment trains multiple models on the ModelNet pointcloud and 
    evaluates them.
    Models used:
        - ECTLinear
        - ECTCNN
    Number of points sampled from the meshes: 
        - 100 
        - 1000 
        - 5000
    """
    experiment = "./experiment/modelnet_points100_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001)

    # Create linear model config
    linear_model = ModelConfig(
            name="ECTLinearModel",
            config = ECTLinearModelConfig(bump_steps=20, hidden=10, num_thetas=30))

    # Create CNN model config
    cnn_model = ModelConfig(
        name="ECTCNNModel",
        config = ECTCNNModelConfig(bump_steps=20, num_thetas=30))

    for samplepoints in [100,1000,5000]: 
        # Create data config
        data = DataModuleConfig(
            name="ModelNetPointsDataModule",
            config=ModelnetConfig(samplepoints=samplepoints,root=f"./data/modelnet{samplepoints}"))
        
        linear_config = Config(data,linear_model,trainer)
        cnn_config = Config(data,cnn_model,trainer)
        
        save_config(linear_config,os.path.join(experiment,f"linear_{samplepoints}.yaml"))
        save_config(cnn_config,os.path.join(experiment,f"cnn_{samplepoints}.yaml"))


def torus_vs_spheres_vs_mobius():
    """
    This experiment trains a ect cnn and linear model to distinguish 
    three classes, a noisy torus,sphere and mobius strip.
    Models used:
        - ECTLinear
        - ECTCNN
    """
    experiment = "./experiment/manifold_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig()

    # Create linear model config
    linear_model = ModelConfig(
            name="ECTLinearModel",
            config = ECTLinearModelConfig(bump_steps=20, hidden=10, num_thetas=30))

    # Create CNN model config
    cnn_model = ModelConfig(
        name="ECTCNNModel",
        config = ECTCNNModelConfig(bump_steps=20, num_thetas=30))

    # Create Data config
    data = DataModuleConfig(
        name="ManifoldDataModule",
        config=ManifoldConfig())
    
    linear_config = Config(data,linear_model,trainer)
    cnn_config = Config(data,cnn_model,trainer)
    
    save_config(linear_config,os.path.join(experiment,f"linear.yaml"))
    save_config(cnn_config,os.path.join(experiment,f"cnn.yaml"))

if __name__ == "__main__":
    theta_sweep()
    modelnet_classification()
    torus_vs_spheres_vs_mobius()







#cnn_theta_sweep()
""" linear_theta_sweep() """
""" model_net_mesh() """

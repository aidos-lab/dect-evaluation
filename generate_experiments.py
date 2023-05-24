import os 
import yaml
from omegaconf import OmegaConf
import itertools
import glob

"""
This script creates all the configurations in the config folder. 
It allows for better reproducebility. The script takes 
NOTE: Gets ran every time make run is called.
"""

data_base_str = """
data:
  name: GNNBenchmarkDataModule
  config:
    name: MNIST
    root: ./data
    split: train
    batch_size: 4
    num_workers: 4
"""

model_base_str = """
model:
  name: ECTCNNModel
  config:
    bump_steps : 20
    num_thetas : 30
    num_features : 3
    R : 1.5
    scale : 500
    num_classes: 10
"""

defaults_base_str = """
trainer:
  lr: 0.001
  num_epochs : 200
"""


def cnn_theta_bump_step_sweep(data_base_str,model_base_str,defaults_base_str):
    experiment_folder = 'cnn_theta_bumpstep_sweep'
    data_base = OmegaConf.create(data_base_str)
    model_base = OmegaConf.create(model_base_str)
    defaults_base = OmegaConf.create(defaults_base_str)

    files = glob.glob(f"./config/{experiment_folder}/*")
    for f in files:
        os.remove(f)

    bump_steps_sweep = [i for i in range(5,20,5)]
    num_thetas_sweep = [i for i in range(5,50,5)]

    for idx, (num_thetas, bump_steps) in enumerate(itertools.product(num_thetas_sweep,bump_steps_sweep)): 
        model_base.model.config.num_thetas = num_thetas
        model_base.model.config.bump_steps = bump_steps
        conf = OmegaConf.merge(data_base,model_base,defaults_base)
        with open(f"./config/{experiment_folder}/{idx}.yaml","w") as f:
            OmegaConf.save(conf,f)


def linear_theta_bump_step_sweep(data_base_str,model_base_str,defaults_base_str):
    experiment_folder = "linear_theta_bumpstep_sweep"
    data_base = OmegaConf.create(data_base_str)
    model_base = OmegaConf.create(model_base_str)
    defaults_base = OmegaConf.create(defaults_base_str)

    files = glob.glob(f"./config/{experiment_folder}/*")
    for f in files:
        os.remove(f)

    bump_steps_sweep = [i for i in range(5,20,5)]
    num_thetas_sweep = [i for i in range(5,50,5)]

    for idx, (num_thetas, bump_steps) in enumerate(itertools.product(num_thetas_sweep,bump_steps_sweep)): 
        model_base.model.name = "ECTPointsLinearModel"
        model_base.model.config.hidden = 10 
        model_base.model.config.num_thetas = num_thetas
        model_base.model.config.bump_steps = bump_steps
        conf = OmegaConf.merge(data_base,model_base,defaults_base)
        with open(f"./config/{experiment_folder}/{idx}.yaml","w") as f:
            OmegaConf.save(conf,f)



def model_net_mesh(data_base_str,model_base_str,defaults_base_str):
    experiment_folder = "modelnet_mesh"
    data_base = OmegaConf.create(data_base_str)
    model_base = OmegaConf.create(model_base_str)
    defaults_base = OmegaConf.create(defaults_base_str)

    files = glob.glob(f"./config/{experiment_folder}/*")
    for f in files:
        os.remove(f)
    data_base.data.name = "ModelNetMeshDataModule"
    model_base.model.name = "ECTPointsLinearModel"
    model_base.model.config.hidden = 10 
    model_base.model.config.num_thetas = 40
    model_base.model.config.bump_steps = 40
    defaults_base.trainer.num_epochs=100
    defaults_base.trainer.lr = 0.01
    
    conf = OmegaConf.merge(data_base,model_base,defaults_base)
    with open(f"./config/{experiment_folder}/linear_modelnet.yaml","w") as f:
        OmegaConf.save(conf,f)


cnn_theta_bump_step_sweep(data_base_str,model_base_str,defaults_base_str)
linear_theta_bump_step_sweep(data_base_str,model_base_str,defaults_base_str)
model_net_mesh(data_base_str,model_base_str,defaults_base_str)

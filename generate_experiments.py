import os 
import yaml
from omegaconf import OmegaConf
import itertools
import glob

from dataclasses import dataclass
from typing import Protocol

"""
This script creates all the configurations in the config folder. 
It allows for better reproducebility. The script takes 
NOTE: Gets ran every time make run is called.
"""

def clean_folder(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)



def cnn_theta_sweep():
    experiment_folder = "cnn_theta_sweep"
    data_base = OmegaConf.load("./config/gnn_benchmark_data_base.yaml") 
    model_base = OmegaConf.load("./config/ect_model_base.yaml")
    trainer_base = OmegaConf.load("./config/trainer_base.yaml")  
    
    clean_folder(f"./config/{experiment_folder}/*")

    #bump_steps_sweep = [i for i in range(5,20,5)]
    num_thetas_sweep = [i for i in range(5,55,5)]

    # Set base parameters for the 

    for idx, num_thetas in enumerate(num_thetas_sweep): 
        model_base.model.config.num_thetas = num_thetas
        model_base.model.config.bump_steps = 20
        conf = OmegaConf.merge(data_base,model_base,trainer_base)
        with open(f"./config/{experiment_folder}/{idx}.yaml","w") as f:
            OmegaConf.save(conf,f)


def linear_theta_sweep():
    experiment_folder = "linear_theta_bumpstep_sweep"
    
    data_base = OmegaConf.load("./config/gnn_benchmark_data_base.yaml") 
    model_base = OmegaConf.load("./config/linear_model_base.yaml")
    trainer_base = OmegaConf.load("./config/trainer_base.yaml")  

    clean_folder(f"./config/{experiment_folder}/*")

    bump_steps_sweep = [i for i in range(5,20,5)]
    num_thetas_sweep = [i for i in range(5,50,5)]

    for idx, (num_thetas, bump_steps) in enumerate(itertools.product(num_thetas_sweep,bump_steps_sweep)): 
        model_base.model.name = "ECTPointsLinearModel"
        model_base.model.config.hidden = 10 
        model_base.model.config.num_thetas = num_thetas
        model_base.model.config.bump_steps = bump_steps
        conf = OmegaConf.merge(data_base,model_base,trainer_base)
        with open(f"./config/{experiment_folder}/{idx}.yaml","w") as f:
            OmegaConf.save(conf,f)



def model_net_mesh():
    experiment_folder = "modelnet_mesh"
    data_base = OmegaConf.load("./config/gnn_benchmark_data_base.yaml") 
    model_base = OmegaConf.load("./config/ect_model_base.yaml")
    trainer_base = OmegaConf.load("./config/trainer_base.yaml")  

    clean_folder(f"./config/{experiment_folder}/*")
    
    data_base.data.name = "ModelNetMeshDataModule"
    model_base.model.name = "ECTPointsLinearModel"
    model_base.model.config.hidden = 10 
    model_base.model.config.num_thetas = 40
    model_base.model.config.bump_steps = 40
    trainer_base.trainer.num_epochs=100
    trainer_base.trainer.lr = 0.01
    
    conf = OmegaConf.merge(data_base,model_base,trainer_base)
    with open(f"./config/{experiment_folder}/linear_modelnet.yaml","w") as f:
        OmegaConf.save(conf,f)


cnn_theta_sweep()
""" linear_theta_sweep() """
""" model_net_mesh() """

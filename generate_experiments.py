import os 
from omegaconf import OmegaConf
import shutil
from config import *

"""
This script creates all the configurations in the config folder. 
It allows for better reproducebility. The script takes 
NOTE: Gets ran every time make run is called.
"""


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
                name="ECTLinearPointsModel",
                config = ECTLinearModelConfig(bump_steps=20, hidden=10, num_thetas=num_thetas))

        cnn_model = ModelConfig(
                name="ECTCNNPointsModel",
                config = ECTCNNModelConfig(bump_steps=20, num_thetas=num_thetas))

        linear_config = Config(data,linear_model,trainer)
        cnn_config = Config(data,cnn_model,trainer)

        save_config(linear_config,os.path.join(linear_experiment,f"{idx}.yaml"))
        save_config(cnn_config,os.path.join(cnn_experiment,f"{idx}.yaml"))



def simple_modelnet_classification():
    """
    This experiment trains two models on the ModelNet pointcloud and 
    evaluates them.
    Models used:
        - ECTLinear
        - ECTCNN
    """
    experiment = "./experiment/modelnet_simple"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001,num_epochs=500)

    # Create CNN model config
    cnn_model = ModelConfig(
        name="ECTCNNPointsModel",
        config = ECTCNNModelConfig(bump_steps=32, 
                                   num_thetas=32,
                                   batch_size=128))

    for samplepoints in [5000]: 
        # Create data config
        data = DataModuleConfig(
            name="ModelNetPointsDataModule",
            config=ModelnetConfig(
                samplepoints=samplepoints,
                root=f"./data/modelnet{samplepoints}",
                num_workers=0,
                batch_size=64
                ))
        
        """ linear_config = Config(data,linear_model,trainer) """
        cnn_config = Config(data,cnn_model,trainer)
        
        """ save_config(linear_config,os.path.join(experiment,f"linear_{samplepoints}.yaml")) """
        save_config(cnn_config,os.path.join(experiment,f"cnn_{samplepoints}.yaml"))


def simple_modelnet40_classification():
    experiment = "./experiment/modelnet40_simple"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001,num_epochs=100)

    # Create CNN model config
    cnn_model = ModelConfig(
        name="ECTCNNPointsModel",
        config = ECTCNNModelConfig(bump_steps=32, 
                                   num_thetas=32,
                                   batch_size=128,
                                   num_classes = 40))

    for samplepoints in [1000]: 
        # Create data config
        data = DataModuleConfig(
            name="ModelNetPointsDataModule",
            config=ModelnetConfig(
                samplepoints=samplepoints,
                root=f"./data/modelnet40{samplepoints}",
                num_workers=0,
                batch_size=64,
                name="40"
                ))
        
        """ linear_config = Config(data,linear_model,trainer) """
        cnn_config = Config(data,cnn_model,trainer)
        
        """ save_config(linear_config,os.path.join(experiment,f"linear_{samplepoints}.yaml")) """
        save_config(cnn_config,os.path.join(experiment,f"cnn_{samplepoints}.yaml"))

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
            name="ECTLinearPointsModel",
            config = ECTLinearModelConfig(bump_steps=20, hidden=10, num_thetas=30))

    # Create CNN model config
    cnn_model = ModelConfig(
        name="ECTCNNPointsModel",
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

def gnn_benchmark_mnist_classification():
    """
    This experiment trains multiple models on the GNN MNIST pointcloud and 
    evaluates them.
    Models used:
        - ECTLinear
        - ECTCNN
    """
    experiment = "./experiment/gnnmnist_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001,num_epochs=200)


    # Create CNN model config
    cnn_pointsmodel = ModelConfig(
        name="ECTCNNPointsModel",
        config = ECTCNNModelConfig(
            num_features=3,
            bump_steps=32, 
            num_thetas=32,
            num_classes=10
            )
        )

    # Create CNN Edges model config
    cnn_edgesmodel = ModelConfig(
        name="ECTCNNEdgesModel",
        config = ECTCNNModelConfig(
            num_features=3,
            bump_steps=32, 
            num_thetas=32,
            num_classes=10
            )
        )


    # Create linear points model config
    linear_pointsmodel = ModelConfig(
            name="ECTLinearPointsModel",
            config = ECTLinearModelConfig(
                num_features=3,
                bump_steps=32, 
                hidden=10, 
                num_thetas=32,
                num_classes=10
                )
            )

    # Create linear edges model config
    linear_edgesmodel = ModelConfig(
            name="ECTLinearEdgesModel",
            config = ECTLinearModelConfig(
                num_features = 3,
                bump_steps = 32, 
                hidden = 10, 
                num_thetas = 32,
                num_classes = 10
                )
            )


    data = DataModuleConfig(
        name="GNNBenchmarkDataModule",
        config=GNNBenchmarkConfig(
            name="MNIST",
            num_workers=12,
            batch_size=128
            )
        )
    
    cnn_points_config = Config(data,cnn_pointsmodel,trainer)
    cnn_edges_config = Config(data,cnn_edgesmodel,trainer)
    linear_points_config = Config(data,linear_pointsmodel,trainer)
    linear_edges_config = Config(data,linear_edgesmodel,trainer)
    
    save_config(cnn_points_config,os.path.join(experiment,f"cnn_points_config.yaml"))
    save_config(cnn_edges_config,os.path.join(experiment,f"cnn_edges_config.yaml"))
    save_config(linear_points_config,os.path.join(experiment,f"linear_points_config.yaml"))
    save_config(linear_edges_config,os.path.join(experiment,f"linear_edges_config.yaml"))





def gnn_benchmark_cifar10_classification():
    """
    This experiment trains multiple models on the GNN CIFAR10 pointcloud and 
    evaluates them.
    Models used:
        - ECTCNN
    """
    experiment = "./experiment/gnncifar10_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.001,num_epochs=100)
    # Create CNN model config
    cnn_model = ModelConfig(
        name="ECTCNNPointsModel",
        config = ECTCNNModelConfig(num_features=5,bump_steps=32, num_thetas=64))
    data = DataModuleConfig(
        name="GNNBenchmarkDataModule",
        config=GNNBenchmarkConfig(name="CIFAR10",
                                  num_workers=12,
                                  batch_size=256))
    cnn_config = Config(data,cnn_model,trainer)
    save_config(cnn_config,os.path.join(experiment,f"cnn.yaml"))

def tu_letter_high_classification():
    """
    This experiment trains multiple models on the GNN CIFAR10 pointcloud and 
    evaluates them.
    Models used:
        - ECTCNN
    """
    experiment = "./experiment/letter_high_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001,num_epochs=700)
    
    # Create CNN model config
    cnn_pointsmodel = ModelConfig(
        name="ECTCNNPointsModel",
        config = ECTCNNModelConfig(num_features=2,bump_steps=32, num_thetas=32,num_classes=15))

    # Create CNN Edges model config
    cnn_edgesmodel = ModelConfig(
        name="ECTCNNEdgesModel",
        config = ECTCNNModelConfig(num_features=2,bump_steps=32, num_thetas=32,num_classes=15))


    # Create linear points model config
    linear_pointsmodel = ModelConfig(
            name="ECTLinearPointsModel",
            config = ECTLinearModelConfig(num_features=2,bump_steps=32, hidden=10, num_thetas=32,num_classes=15))

    # Create linear edges model config
    linear_edgesmodel = ModelConfig(
            name="ECTLinearEdgesModel",
            config = ECTLinearModelConfig(num_features=2,bump_steps=32, hidden=10, num_thetas=32,num_classes=15))


    data = DataModuleConfig(
        name="TUDataModule",
            config=TUDataConfig(name="Letter-high",
                  num_workers=0,
                  batch_size=64))

    cnn_points_config = Config(data,cnn_pointsmodel,trainer)
    cnn_edges_config = Config(data,cnn_edgesmodel,trainer)
    linear_points_config = Config(data,linear_pointsmodel,trainer)
    linear_edges_config = Config(data,linear_pointsmodel,trainer)
    
    save_config(cnn_points_config,os.path.join(experiment,f"cnn_points_config.yaml"))
    save_config(cnn_edges_config,os.path.join(experiment,f"cnn_edges_config.yaml"))
    save_config(linear_points_config,os.path.join(experiment,f"linear_points_config.yaml"))
    save_config(linear_edges_config,os.path.join(experiment,f"linear_edges_config.yaml"))






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
            name="ECTLinearPointsModel",
            config = ECTLinearModelConfig(bump_steps=32, hidden=10, num_thetas=32))

    # Create CNN model config
    cnn_model = ModelConfig(
        name="ECTCNNPointsModel",
        config = ECTCNNModelConfig(bump_steps=20, num_thetas=30))

    # Create Data config
    data = DataModuleConfig(
        name="ManifoldDataModule",
        config=ManifoldConfig(pin_memory=True))
    
    linear_config = Config(data,linear_model,trainer)
    cnn_config = Config(data,cnn_model,trainer)
    
    save_config(linear_config,os.path.join(experiment,f"linear.yaml"))
    save_config(cnn_config,os.path.join(experiment,f"cnn.yaml"))


""" def gnn_benchmark_cifar10_classification(): """
"""     """ """
"""     """ """
"""     experiment = "./experiment/gnncifar10_classification" """
"""     create_experiment_folder(experiment) """
""""""
"""     # Create Trainer Config """
"""     trainer = TrainerConfig(lr=0.001,num_epochs=100) """
"""     # Create CNN model config """
"""     cnn_model = ModelConfig( """
"""         name="ECTCNNPointsModel", """
"""         config = ECTCNNModelConfig(num_features=5,bump_steps=32, num_thetas=64)) """
"""     data = DataModuleConfig( """
"""         name="GNNBenchmarkDataModule", """
"""         config=GNNBenchmarkConfig(name="CIFAR10", """
"""                                   num_workers=12, """
"""                                   batch_size=256)) """
"""     cnn_config = Config(data,cnn_model,trainer) """
"""     save_config(cnn_config,os.path.join(experiment,f"cnn.yaml")) """
""""""




if __name__ == "__main__":
    theta_sweep()
    modelnet_classification()
    torus_vs_spheres_vs_mobius()
    gnn_benchmark_mnist_classification()
    gnn_benchmark_cifar10_classification()
    simple_modelnet_classification()
    tu_letter_high_classification()
    simple_modelnet40_classification()







#cnn_theta_sweep()
""" linear_theta_sweep() """
""" model_net_mesh() """

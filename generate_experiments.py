import os
from omegaconf import OmegaConf
import shutil
from datasets.tu import TUDataModuleConfig
from models.base_model import ModelConfig

from config import Config, TrainerConfig, Meta

"""
This script creates all the configurations in the config folder. 
It allows for better reproducebility. The script takes 
NOTE: Gets ran every time make run is called.
"""


#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper methods                                           │
#  ╰──────────────────────────────────────────────────────────╯


def create_experiment_folder(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def save_config(cfg, path):
    c = OmegaConf.create(cfg)
    with open(path, "w") as f:
        OmegaConf.save(c, f)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Experiments                                              │
#  ╰──────────────────────────────────────────────────────────╯


def tu_letter_high_classification() -> None:
    """
    This experiment trains and classifies the letter high dataset in
    the TU dataset.
    """

    experiment = "./experiment/letter_high_classification"
    create_experiment_folder(experiment)

    # Create meta data
    meta = Meta("desct-test-new")

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=100)

    # Base configs for the models, same for all in this
    # experiment
    num_features = 2
    bump_steps = 32
    num_thetas = 32
    num_classes = 15

    # Create CNN model config
    ect_cnn_points = ModelConfig(
        module="models.ect_cnn_points",
        num_features=num_features,
        bump_steps=bump_steps,
        num_thetas=num_thetas,
        num_classes=num_classes,
        hidden=100,
    )

    # Create CNN Edges model config
    ect_cnn_edges = ModelConfig(
        module="models.ect_cnn_edges",
        num_features=num_features,
        bump_steps=bump_steps,
        num_thetas=num_thetas,
        num_classes=num_classes,
        hidden=100,
    )

    # Create linear points model config
    ect_linear_points = ModelConfig(
        module="models.ect_linear_points",
        num_features=num_features,
        bump_steps=bump_steps,
        num_thetas=num_thetas,
        num_classes=num_classes,
        hidden=100,
    )

    # Create linear edges model config
    ect_linear_edges = ModelConfig(
        module="models.ect_linear_edges",
        num_features=num_features,
        bump_steps=bump_steps,
        num_thetas=num_thetas,
        num_classes=num_classes,
        hidden=100,
    )

    # Create the dataset config.
    data = TUDataModuleConfig(
        module="datasets.tu",
        root="./data",
        name="Letter-high",
        num_workers=0,
        batch_size=64,
        pin_memory=True,
    )

    cnn_points_config = Config(meta, data, ect_cnn_points, trainer)
    cnn_edges_config = Config(meta, data, ect_cnn_edges, trainer)
    linear_points_config = Config(meta, data, ect_linear_points, trainer)
    linear_edges_config = Config(meta, data, ect_linear_edges, trainer)

    save_config(cnn_points_config, os.path.join(experiment, f"cnn_points_config.yaml"))
    save_config(cnn_edges_config, os.path.join(experiment, f"cnn_edges_config.yaml"))
    save_config(
        linear_points_config, os.path.join(experiment, f"linear_points_config.yaml")
    )
    save_config(
        linear_edges_config, os.path.join(experiment, f"linear_edges_config.yaml")
    )


def gnn_mnist_classification() -> None:
    """
    This experiment trains multiple models on the GNN MNIST pointcloud and
    evaluates them.
    Models used:
        - ECTLinear
        - ECTCNN
    """
    experiment = "./experiment/gnn_mnist_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.001, num_epochs=100)

    # Base configs for the models, same for all in this
    # experiment
    num_features = 3
    bump_steps = 32
    num_thetas = 32
    num_classes = 10
    hidden = 100

    # Create CNN model config
    cnn_pointsmodel = ModelConfig(
        name="ECTCNNPointsModel",
        config=ECTCNNModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=hidden,
        ),
    )

    # Create CNN Edges model config
    cnn_edgesmodel = ModelConfig(
        name="ECTCNNEdgesModel",
        config=ECTCNNModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=hidden,
        ),
    )

    # Create linear points model config
    linear_pointsmodel = ModelConfig(
        name="ECTLinearPointsModel",
        config=ECTLinearModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=100,
        ),
    )

    # Create linear edges model config
    linear_edgesmodel = ModelConfig(
        name="ECTLinearEdgesModel",
        config=ECTLinearModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=100,
        ),
    )

    # Create the dataset config.
    data = DataModuleConfig(
        name="GNNBenchmarkDataModule",
        config=GNNBenchmarkConfig(name="MNIST", num_workers=4, batch_size=128),
    )

    cnn_points_config = Config(data, cnn_pointsmodel, trainer)
    cnn_edges_config = Config(data, cnn_edgesmodel, trainer)
    linear_points_config = Config(data, linear_pointsmodel, trainer)
    linear_edges_config = Config(data, linear_edgesmodel, trainer)

    save_config(cnn_points_config, os.path.join(experiment, f"cnn_points_config.yaml"))
    save_config(cnn_edges_config, os.path.join(experiment, f"cnn_edges_config.yaml"))
    save_config(
        linear_points_config, os.path.join(experiment, f"linear_points_config.yaml")
    )
    save_config(
        linear_edges_config, os.path.join(experiment, f"linear_edges_config.yaml")
    )


def gnn_cifar10_classification() -> None:
    """
    This experiment trains multiple models on the GNN CIFAR10 super pixel
    graph and evaluates them.
    Models used:
        - Full ECT + CNN
        - Full ECT + Linear
        - Points ECT + CNN
        - Points ECT + Linear
    """
    experiment = "./experiment/gnncifar10_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.001, num_epochs=100)

    # Base configs for the models, same for all in this
    # experiment
    num_features = 5
    bump_steps = 32
    num_thetas = 32
    num_classes = 10

    # Create CNN model config
    cnn_pointsmodel = ModelConfig(
        name="ECTCNNPointsModel",
        config=ECTCNNModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
        ),
    )

    # Create CNN Edges model config
    cnn_edgesmodel = ModelConfig(
        name="ECTCNNEdgesModel",
        config=ECTCNNModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
        ),
    )

    # Create linear points model config
    linear_pointsmodel = ModelConfig(
        name="ECTLinearPointsModel",
        config=ECTLinearModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=100,
        ),
    )

    # Create linear edges model config
    linear_edgesmodel = ModelConfig(
        name="ECTLinearEdgesModel",
        config=ECTLinearModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=100,
        ),
    )

    # Create the dataset config.
    data = DataModuleConfig(
        name="GNNBenchmarkDataModule",
        config=GNNBenchmarkConfig(name="CIFAR10", num_workers=12, batch_size=256),
    )

    cnn_points_config = Config(data, cnn_pointsmodel, trainer)
    cnn_edges_config = Config(data, cnn_edgesmodel, trainer)
    linear_points_config = Config(data, linear_pointsmodel, trainer)
    linear_edges_config = Config(data, linear_edgesmodel, trainer)

    save_config(cnn_points_config, os.path.join(experiment, f"cnn_points_config.yaml"))
    save_config(cnn_edges_config, os.path.join(experiment, f"cnn_edges_config.yaml"))
    save_config(
        linear_points_config, os.path.join(experiment, f"linear_points_config.yaml")
    )
    save_config(
        linear_edges_config, os.path.join(experiment, f"linear_edges_config.yaml")
    )


def gnn_modelnet10_classification() -> None:
    """
    This experiment trains multiple models on the GNN Modelnet10 classification
    using the following models
    Models used:
        - Full ECT + CNN
        - Full ECT + Linear
        - Points ECT + CNN
        - Points ECT + Linear
    """
    experiment = "./experiment/gnn_modelnet10_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=100)

    # Base configs for the models, same for all in this
    # Experiment
    num_features = 3
    bump_steps = 32
    num_thetas = 32
    num_classes = 10
    hidden = 100

    # Create CNN model config
    cnn_pointsmodel = ModelConfig(
        name="ECTCNNPointsModel",
        config=ECTCNNModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=hidden,
        ),
    )

    # Create linear points model config
    linear_pointsmodel = ModelConfig(
        name="ECTLinearPointsModel",
        config=ECTLinearModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=hidden,
        ),
    )

    for samplepoints in [100, 1000, 5000]:
        # Create the dataset configuration.
        data = DataModuleConfig(
            name="ModelNetPointsDataModule",
            config=ModelnetConfig(
                samplepoints=samplepoints,
                root=f"./data/modelnet{samplepoints}",
                batch_size=64,
                num_workers=0,
            ),
        )

        cnn_points_config = Config(data, cnn_pointsmodel, trainer)
        linear_points_config = Config(data, linear_pointsmodel, trainer)

        save_config(
            cnn_points_config,
            os.path.join(experiment, f"cnn_points_{samplepoints}_config.yaml"),
        )
        save_config(
            linear_points_config,
            os.path.join(experiment, f"linear_points_{samplepoints}_config.yaml"),
        )


def gnn_modelnet40_classification() -> None:
    """
    This experiment trains multiple models on the GNN Modelnet10 classification
    using the following models
    Models used:
        - Full ECT + CNN
        - Full ECT + Linear
        - Points ECT + CNN
        - Points ECT + Linear
    """
    experiment = "./experiment/gnn_modelnet40_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.00001, num_epochs=100)

    # Base configs for the models, same for all in this
    # experiment
    num_features = 3
    bump_steps = 32
    num_thetas = 32
    num_classes = 40

    # Create CNN model config
    cnn_pointsmodel = ModelConfig(
        name="ECTCNNPointsModel",
        config=ECTCNNModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=800,
        ),
    )

    # Create linear points model config
    linear_pointsmodel = ModelConfig(
        name="ECTLinearPointsModel",
        config=ECTLinearModelConfig(
            num_features=num_features,
            bump_steps=bump_steps,
            num_thetas=num_thetas,
            num_classes=num_classes,
            hidden=500,
        ),
    )

    for samplepoints in [100]:
        # Create the dataset configuration.
        data = DataModuleConfig(
            name="ModelNetPointsDataModule",
            config=ModelnetConfig(
                name="40",
                samplepoints=samplepoints,
                root=f"./data/modelnet40{samplepoints}",
                batch_size=256,
            ),
        )

        cnn_points_config = Config(data, cnn_pointsmodel, trainer)
        linear_points_config = Config(data, linear_pointsmodel, trainer)

        save_config(
            cnn_points_config,
            os.path.join(experiment, f"cnn_points_{samplepoints}_config.yaml"),
        )
        save_config(
            linear_points_config,
            os.path.join(experiment, f"linear_points_{samplepoints}_config.yaml"),
        )


def manifold_classification() -> None:
    """
    This experiment trains a ect cnn and linear model to distinguish
    three classes,
    - a noisy torus,
    - a sphere and
    - a mobius strip.

    Models used:
        - ECTLinear
        - ECTCNN
    """

    experiment = "./experiment/manifold_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(num_epochs=100, lr=0.0001)

    # Create linear model config
    linear_pointsmodel = ModelConfig(
        name="ECTLinearPointsModel",
        config=ECTLinearModelConfig(
            bump_steps=32, hidden=10, num_thetas=32, num_classes=3
        ),
    )

    # Create linear model config
    linear_edgesmodel = ModelConfig(
        name="ECTLinearEdgesModel",
        config=ECTLinearModelConfig(
            bump_steps=32, hidden=10, num_thetas=32, num_classes=3
        ),
    )

    # Create CNN model config
    cnn_pointsmodel = ModelConfig(
        name="ECTCNNPointsModel",
        config=ECTCNNModelConfig(bump_steps=32, num_thetas=32, num_classes=3),
    )

    # Create CNN model config
    cnn_edgesmodel = ModelConfig(
        name="ECTCNNEdgesModel",
        config=ECTCNNModelConfig(bump_steps=32, num_thetas=32, num_classes=3),
    )

    # Create Data config
    data = DataModuleConfig(
        name="ManifoldDataModule",
        config=ManifoldConfig(pin_memory=True, batch_size=64, num_samples=100),
    )

    linear_points_config = Config(data, linear_pointsmodel, trainer)
    linear_edges_config = Config(data, linear_edgesmodel, trainer)
    cnn_points_config = Config(data, cnn_pointsmodel, trainer)
    cnn_edges_config = Config(data, cnn_edgesmodel, trainer)

    save_config(linear_points_config, os.path.join(experiment, f"linear_points.yaml"))
    save_config(linear_edges_config, os.path.join(experiment, f"linear_edges.yaml"))
    save_config(cnn_points_config, os.path.join(experiment, f"cnn_points.yaml"))
    save_config(cnn_edges_config, os.path.join(experiment, f"cnn_edges.yaml"))


def theta_sweep():
    """
    This experiment trains two models with varying number of angles used.
    The configs are stored in separate folders so the two experiments
    can be ran independently (it takes a while).

    The models are trained on the mnist super pixel dataset
    """

    linear_experiment = "./experiment/linear_theta_sweep"
    create_experiment_folder(linear_experiment)

    cnn_experiment = "./experiment/cnn_theta_sweep"
    create_experiment_folder(cnn_experiment)

    # Create the dataset config.
    data = DataModuleConfig(
        name="GNNBenchmarkDataModule",
        config=GNNBenchmarkConfig(name="MNIST", num_workers=4, batch_size=128),
    )

    # Create Trainer Config
    trainer = TrainerConfig(num_epochs=100, lr=0.001)

    """ for idx, num_thetas in [[8,45],[9,50]]:  """

    for idx, num_thetas in enumerate(range(5, 55, 5)):
        linear_model = ModelConfig(
            name="ECTLinearPointsModel",
            config=ECTLinearModelConfig(
                bump_steps=32, hidden=100, num_thetas=num_thetas, num_classes=10
            ),
        )

        cnn_model = ModelConfig(
            name="ECTCNNPointsModel",
            config=ECTCNNModelConfig(
                bump_steps=32, hidden=100, num_thetas=num_thetas, num_classes=10
            ),
        )

        linear_config = Config(data, linear_model, trainer)
        cnn_config = Config(data, cnn_model, trainer)

        save_config(
            linear_config, os.path.join(linear_experiment, f"linear_{idx}.yaml")
        )
        save_config(cnn_config, os.path.join(cnn_experiment, f"cnn_{idx}.yaml"))


if __name__ == "__main__":
    tu_letter_high_classification()
    """ gnn_mnist_classification() """
    """ gnn_cifar10_classification() """
    """ gnn_modelnet10_classification() """
    """ gnn_modelnet40_classification() """
    """ manifold_classification() """
    """ theta_sweep() """

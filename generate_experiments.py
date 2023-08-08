import os
from omegaconf import OmegaConf
import shutil
from datasets.tu import TUDataModuleConfig
from datasets.gnn_benchmark import GNNBenchmarkDataModuleConfig
from datasets.modelnet import ModelNetDataModuleConfig
from datasets.manifold import ManifoldDataModuleConfig


from models.base_model import ECTModelConfig
from config import Config, TrainerConfig, Meta

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

# Create meta data
meta = Meta("desct-test-new")

# Create Trainer Config
trainer = TrainerConfig(lr=0.0001, num_epochs=100)


def tu_letter_high_classification() -> None:
    experiment = "./experiment/letter_high_classification"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_cnn_points",
        "models.ect_cnn_edges",
        "models.ect_linear_points",
        "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TUDataModuleConfig(
        module="datasets.tu",
        name="Letter-high",
        num_workers=0,
    )

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module,
            num_features=2,
            num_classes=15,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def gnn_classification(name="MNIST") -> None:
    experiment = f"./experiment/gnn_{name.lower()}_classification"
    create_experiment_folder(experiment)

    # Create meta data
    meta = Meta("desct-test-new")

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=100)

    modules = [
        "models.ect_cnn_points",
        "models.ect_cnn_edges",
        "models.ect_linear_points",
        "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = GNNBenchmarkDataModuleConfig(
        module="datasets.gnn_benchmark",
        name=name,
    )

    for module in modules:
        # Create linear points model config
        modelconfig = ECTModelConfig(module=module)

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def gnn_modelnet_classification(name="10") -> None:
    experiment = f"./experiment/gnn_modelnet_{name}_classification"
    create_experiment_folder(experiment)

    # Create meta data
    meta = Meta("desct-test-new")

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=100)

    modules = [
        "models.ect_cnn_points",
        "models.ect_linear_points",
    ]

    for module in modules:
        for samplepoints in [100, 1000, 5000]:
            # Create linear points model config
            modelconfig = ECTModelConfig(module=module)

            # Create the dataset config.
            data = ModelNetDataModuleConfig(
                module="datasets.modelnet",
                samplepoints=samplepoints,
            )

            config = Config(meta, data, modelconfig, trainer)
            save_config(
                config,
                os.path.join(
                    experiment, f"{module.split(sep='.')[1]}_{samplepoints}.yaml"
                ),
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

    # Create meta data
    meta = Meta("desct-test-new")

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=100)

    modules = [
        "models.ect_cnn_points",
        "models.ect_cnn_edges",
        "models.ect_linear_points",
        "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = ManifoldDataModuleConfig(
        module="datasets.manifold",
    )

    for module in modules:
        # Create linear points model config
        modelconfig = ECTModelConfig(module=module)

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def theta_sweep():
    """
    This experiment trains two models with varying number of angles used.
    The configs are stored in separate folders so the two experiments
    can be ran independently (it takes a while).

    The models are trained on the mnist super pixel dataset
    """

    experiment = "./experiment/theta_sweep"
    create_experiment_folder(experiment)

    # Create meta data
    meta = Meta("desct-test-new")
    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=100)

    modules = [
        "models.ect_cnn_points",
        "models.ect_cnn_edges",
        "models.ect_linear_points",
        "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = GNNBenchmarkDataModuleConfig(
        module="datasets.gnn_benchmark",
        name="MNIST",
    )

    for module in modules:
        for theta in range(5, 55, 5):
            # Create linear points model config
            modelconfig = ECTModelConfig(
                module=module,
                num_thetas=theta,
            )

            config = Config(meta, data, modelconfig, trainer)
            save_config(
                config,
                os.path.join(experiment, f"{module.split(sep='.')[1]}_{theta}.yaml"),
            )


if __name__ == "__main__":
    tu_letter_high_classification()
    gnn_classification("MNIST")
    gnn_classification("CIFAR10")
    gnn_modelnet_classification("10")
    gnn_modelnet_classification("40")
    manifold_classification()
    theta_sweep()

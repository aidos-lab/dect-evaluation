import os
from omegaconf import OmegaConf
import shutil
from datasets.gnn_benchmark import GNNBenchmarkDataModuleConfig
from datasets.modelnet import ModelNetDataModuleConfig
from datasets.manifold import ManifoldDataModuleConfig
from datasets.ogb import MOLHIVDataModuleConfig, OGBDataModule
from datasets.tu import (
    TUDataModule,
    TUEnzymesConfig,
    TUIMDBBConfig,
    TUDDConfig,
    TUProteinsFullConfig,
    TURedditBConfig,
    TULetterHighConfig,
    TULetterLowConfig,
    TULetterMedConfig,
    TUNCI1Config,
    TUNCI109Config,
)
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


# def tu_bbbp(experiment_folder="experiment", trainer=None, meta=None) -> None:
#     experiment = f"./{experiment_folder}/BBBP"
#     create_experiment_folder(experiment)

#     modules = [
#         # "models.ect_cnn_points",
#         "models.ect_cnn_edges",
#         # "models.ect_linear_points",
#         # "models.ect_linear_edges",
#     ]

#     # Create the dataset config.
#     data = TUNCI1Config()

#     for module in modules:
#         modelconfig = ECTModelConfig(
#             module=module, num_features=30, num_classes=2, num_thetas=32, bump_steps=32
#         )

#         config = Config(meta, data, modelconfig, trainer)
#         save_config(
#             config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
#         )


def tu_nci1(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/nci1"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]
    trainer = TrainerConfig(lr=0.001, num_epochs=100, num_reruns=5)
    # Create the dataset config.
    data = TUNCI1Config()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=30, num_classes=2, num_thetas=32, bump_steps=32
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_nci109(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/nci109"
    create_experiment_folder(experiment)
    trainer = TrainerConfig(lr=0.001, num_epochs=500, num_reruns=5)
    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TUNCI109Config(batch_size=64)

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module,
            num_features=36,
            num_classes=2,
            num_thetas=64,
            bump_steps=64,
            hidden=50,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def ogb_mol(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/OGB-MOLHIV"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = MOLHIVDataModuleConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=9, num_classes=2, num_thetas=32, bump_steps=32
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_reddit_b(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/REDDIT-BINARY"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TURedditBConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=1, num_classes=2, num_thetas=32, bump_steps=32
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_imdb_b(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/IMDB-BINARY"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TUIMDBBConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=541, num_classes=2, num_thetas=32, bump_steps=32
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_letter_low_classification(
    experiment_folder="experiment", trainer=None, meta=None
) -> None:
    experiment = f"./{experiment_folder}/Letter-low"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TULetterLowConfig(batch_size=32)

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=2, num_classes=15, hidden=100
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_letter_med_classification(
    experiment_folder="experiment", trainer=None, meta=None
) -> None:
    experiment = f"./{experiment_folder}/Letter-med"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TULetterMedConfig(batch_size=32)

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=2, num_classes=15, hidden=100
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_letter_high_classification(
    experiment_folder="experiment", trainer=None, meta=None
) -> None:
    experiment = f"./{experiment_folder}/Letter-high"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]
    trainer = TrainerConfig(lr=0.001, num_epochs=150, num_reruns=5)
    # Create the dataset config.
    data = TULetterHighConfig(batch_size=32)

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=2, num_classes=15, hidden=100
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_dd(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/DD"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TUDDConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=89, num_classes=2, num_thetas=32, bump_steps=32
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_enzymes(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/ENZYMES"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TUEnzymesConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=21, num_classes=6, num_thetas=32, bump_steps=32
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_proteins(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/PROTEINS_full"
    create_experiment_folder(experiment)

    trainer = TrainerConfig(lr=0.001, num_epochs=100, num_reruns=5)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = TUProteinsFullConfig(batch_size=128)

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module,
            num_features=32,
            num_classes=2,
            num_thetas=32,
            bump_steps=32,
            hidden=50,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def gnn_classification(
    name="MNIST", experiment_folder="experiment", trainer=None, meta=None
) -> None:
    experiment = f"./{experiment_folder}/gnn_{name.lower()}_classification"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
    ]

    # Create the dataset config.
    data = GNNBenchmarkDataModuleConfig(
        module="datasets.gnn_benchmark",
        batch_size=256,
        name=name,
        pin_memory=False,
    )

    if name == "MNIST" or name == "PATTERN":
        num_features = 3
    else:
        num_features = 5

    if name == "PATTERN":
        num_classes = 2
    else:
        num_classes = 10

    for module in modules:
        # Create linear points model config
        modelconfig = ECTModelConfig(
            module=module,
            num_features=num_features,
            hidden=100,
            num_classes=num_classes,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def gnn_modelnet_classification(
    name="10", experiment_folder="experiment", trainer=None, meta=None
) -> None:
    experiment = f"./{experiment_folder}/gnn_modelnet_{name}_classification"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_cnn_points",
        # "models.ect_linear_points",
    ]

    for module in modules:
        for samplepoints in [100, 1000, 5000]:
            # Create linear points model config
            modelconfig = ECTModelConfig(module=module)

            # Create the dataset config.
            data = ModelNetDataModuleConfig(
                root=f"./data/modelnet_{name}_{samplepoints}",
                module="datasets.modelnet",
                samplepoints=samplepoints,
                name=name,
                drop_last=True,
            )

            config = Config(meta, data, modelconfig, trainer)
            save_config(
                config,
                os.path.join(
                    experiment, f"{module.split(sep='.')[1]}_{samplepoints}.yaml"
                ),
            )


def manifold_classification(
    experiment_folder="experiment", trainer=None, meta=None
) -> None:
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

    experiment = f"./{experiment_folder}/manifold_classification"
    create_experiment_folder(experiment)

    modules = [
        # "models.ect_cnn_points",
        # "models.ect_cnn_edges",
        # "models.ect_linear_points",
        # "models.ect_linear_edges",
        "models.ect_linear_faces",
        "models.ect_cnn_faces",
    ]

    # Create the dataset config.
    data = ManifoldDataModuleConfig(module="datasets.manifold", batch_size=32)

    for module in modules:
        # Create linear points model config
        modelconfig = ECTModelConfig(module=module)

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def theta_sweep(experiment_folder="experiment", trainer=None, meta=None):
    experiment = f"./{experiment_folder}/theta_sweep"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_linear_points",
    ]

    # Create the dataset config.
    data = GNNBenchmarkDataModuleConfig(
        module="datasets.gnn_benchmark", name="MNIST", pin_memory=False
    )

    for module in modules:
        for theta in range(1, 32):
            # Create linear points model config
            modelconfig = ECTModelConfig(module=module, num_thetas=theta, hidden=100)

            config = Config(meta, data, modelconfig, trainer)
            save_config(
                config,
                os.path.join(
                    experiment, f"{module.split(sep='.')[1]}_{theta:03d}.yaml"
                ),
            )


if __name__ == "__main__":
    # Create Trainer Config

    trainer = TrainerConfig(lr=0.01, num_epochs=100, num_reruns=5)
    # Create meta data
    meta = Meta("desct-test-new")
    experiment_folder = "experiment"
    tu_nci1(experiment_folder, trainer, meta)
    tu_nci109(experiment_folder, trainer, meta)
    # tu_proteins(experiment_folder, trainer, meta)
    # tu_dd(experiment_folder, trainer, meta)
    # tu_enzymes(experiment_folder, trainer, meta)
    # tu_imdb_b(experiment_folder, trainer, meta)
    # tu_reddit_b(experiment_folder, trainer, meta)
    # tu_letter_high_classification(experiment_folder, trainer, meta)
    # tu_letter_med_classification(experiment_folder, trainer, meta)
    # tu_letter_low_classification(experiment_folder, trainer, meta)

    # ogb_mol(experiment_folder, trainer, meta)
    # gnn_classification("MNIST", experiment_folder, trainer, meta)
    # gnn_classification("CIFAR10", experiment_folder, trainer, meta)
    # gnn_modelnet_classification("10", experiment_folder, trainer, meta)
    # gnn_modelnet_classification("40", experiment_folder, trainer, meta)
    # manifold_classification(experiment_folder, trainer, meta)
    # theta_sweep(experiment_folder, trainer, meta)

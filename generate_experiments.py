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
    TUBZRConfig,
    TUCOX2Config,
    TUFrankensteinConfig,
    TUFingerprintConfig,
    TUDHFRConfig,
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
<<<<<<< HEAD

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Experiments                                              │
    #  ╰──────────────────────────────────────────────────────────╯
=======
>>>>>>> main


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

# ,
# TUFrankensteinConfig,
# TUFingerprintConfig,
# TUDHFRConfig,


<<<<<<< HEAD
def tu_dhfr(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/dhfr"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_cnn_edges",
    ]
    trainer = TrainerConfig(lr=0.001, num_epochs=200, num_reruns=5)
    # Create the dataset config.
    data = TUDHFRConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module,
            num_features=38,
            num_classes=2,
            num_thetas=32,
            bump_steps=32,
            hidden=40,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_fingerprint(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/fingerprint"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_cnn_edges",
    ]
    # trainer = TrainerConfig(lr=0.001, num_epochs=100, num_reruns=5)
    # Create the dataset config.
    data = TUFingerprintConfig(cleaned=False)

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module,
            num_features=2,
            num_classes=6,
            num_thetas=32,
            bump_steps=32,
            hidden=40,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_frankenstein(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/frankenstein"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_cnn_edges",
    ]
    # trainer = TrainerConfig(lr=0.001, num_epochs=100, num_reruns=5)
    # Create the dataset config.
    data = TUFrankensteinConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module,
            num_features=780,
            num_classes=2,
            num_thetas=32,
            bump_steps=32,
            hidden=20,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_cox2(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/cox2"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_cnn_edges",
    ]
    trainer = TrainerConfig(lr=0.001, num_epochs=200, num_reruns=5)
    # Create the dataset config.
    data = TUCOX2Config()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module, num_features=38, num_classes=2, num_thetas=32, bump_steps=32
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


def tu_bzr(experiment_folder="experiment", trainer=None, meta=None) -> None:
    experiment = f"./{experiment_folder}/bzr"
    create_experiment_folder(experiment)

    modules = [
        "models.ect_cnn_edges",
    ]
    trainer = TrainerConfig(lr=0.001, num_epochs=200, num_reruns=5)
    # Create the dataset config.
    data = TUBZRConfig()

    for module in modules:
        modelconfig = ECTModelConfig(
            module=module,
            num_features=38,
            num_classes=2,
            num_thetas=32,
            bump_steps=32,
            hidden=50,
        )

        config = Config(meta, data, modelconfig, trainer)
        save_config(
            config, os.path.join(experiment, f"{module.split(sep='.')[1]}.yaml")
        )


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
=======
def tu_letter_high_classification() -> None:
    """
    This experiment trains and classifies the letter high dataset in
    the TU dataset.
    """

    experiment = "./experiment/letter_high_classification"
    create_experiment_folder(experiment)

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=100)

    # Base configs for the models, same for all in this
    # experiment
    num_features = 2
    bump_steps = 32
    num_thetas = 32
    num_classes = 15

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
        name="TUDataModule",
        config=TUDataConfig(name="Letter-high", num_workers=0, batch_size=64),
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
>>>>>>> main

    cnn_points_config = Config(data, cnn_pointsmodel, trainer)
    cnn_edges_config = Config(data, cnn_edgesmodel, trainer)
    linear_points_config = Config(data, linear_pointsmodel, trainer)
    linear_edges_config = Config(data, linear_edgesmodel, trainer)

<<<<<<< HEAD
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
    trainer = TrainerConfig(lr=0.001, num_epochs=200, num_reruns=5)
    # Create the dataset config.
    data = GNNBenchmarkDataModuleConfig(
        module="datasets.gnn_benchmark",
        batch_size=64,
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
            hidden=200,
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
=======
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
>>>>>>> main


def manifold_classification(
    experiment_folder="experiment", trainer=None, meta=None
) -> None:
    """
<<<<<<< HEAD
=======
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
>>>>>>> main
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

<<<<<<< HEAD
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

    trainer = TrainerConfig(lr=0.001, num_epochs=100, num_reruns=5)
    # Create meta data
    meta = Meta("desct-test-new")
    experiment_folder = "experiment"
    # tu_bzr(experiment_folder, trainer, meta)
    # tu_cox2(experiment_folder, trainer, meta)
    # tu_frankenstein(experiment_folder, trainer, meta)
    # tu_fingerprint(experiment_folder, trainer, meta)
    # tu_dhfr(experiment_folder, trainer, meta)

    # tu_cox2(experiment_folder, trainer, meta)
    # tu_proteins(experiment_folder, trainer, meta)
    # tu_dd(experiment_folder, trainer, meta)
    # tu_imdb_b(experiment_folder, trainer, meta)
    # tu_reddit_b(experiment_folder, trainer, meta)
    # tu_letter_high_classification(experiment_folder, trainer, meta)
    # tu_letter_med_classification(experiment_folder, trainer, meta)
    # tu_letter_low_classification(experiment_folder, trainer, meta)

    # ogb_mol(experiment_folder, trainer, meta)
    # # gnn_classification("MNIST", experiment_folder, trainer, meta)
    # gnn_classification("CIFAR10", experiment_folder, trainer, meta)
    # # gnn_modelnet_classification("10", experiment_folder, trainer, meta)
    # gnn_modelnet_classification("40", experiment_folder, trainer, meta)
    manifold_classification(experiment_folder, trainer, meta)
    # theta_sweep(experiment_folder, trainer, meta)
=======
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
    gnn_mnist_classification()
    gnn_cifar10_classification()
    gnn_modelnet10_classification()
    gnn_modelnet40_classification()
    manifold_classification()
    theta_sweep()
>>>>>>> main

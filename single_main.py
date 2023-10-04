import torch
import os
from utils import listdir
from torch_geometric.loader import DataLoader, ImbalancedSampler
from datasets.gnn_benchmark import CenterTransform, ThresholdTransform
from config import TrainerConfig, Meta, Config
from datasets.gnn_benchmark import GNNBenchmarkDataModuleConfig
from datasets.modelnet import ModelNetPointsDataModule, ModelNetDataModuleConfig
from models.base_model import ECTModelConfig
from main import compute_avg

# from generate_experiments import (
#     create_experiment_folder,
#     tu_letter_high_classification,
#     gnn_classification,
#     gnn_modelnet_classification,
#     manifold_classification,
#     theta_sweep,
#     save_config,
# )z

import torchvision.transforms as transforms

from loaders.factory import load_module
from torch_geometric.datasets import GNNBenchmarkDataset
from main import Experiment
from logger import Logger, timing
import time

mylogger = Logger()


def main():
    accs = []
    for _ in range(5):
        print("Running experiment", "ect_cnn_best.yaml")
        # exp = Experiment(
        #     "./experiment/Letter-high/ect_cnn_edges.yaml", logger=mylogger, dev=True
        # )
        exp = Experiment(
            "./experiment/manifold_classification/ect_cnn_faces.yaml",
            logger=mylogger,
            dev=True,
        )
        loss, acc = exp.run()
        accs.append(acc)
    compute_avg(torch.tensor(accs))


if __name__ == "__main__":
    main()

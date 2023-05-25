import wandb
from logs import log_msg
import glob
from datasets import load_datamodule
from models import load_model
from omegaconf import OmegaConf
import torch
import os
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
This is an almost copy of the main program and runs ALL experiments in the experiment 
folder for 1 epoch to check if everything works. 
It does not interfere with the configs.
"""

from main import run_experiment


def test_all_experiments():
    experiments = os.listdir("./experiment")
    experiments = [
            "modelnet_points100_classification",
            "manifold_classification",
            "cnn_theta_sweep",
            "linear_theta_sweep",
            ]
    for experiment in experiments: 
        print("Running experiment", experiment)
        run_experiment(experiment,dev=True)


if __name__ == "__main__":
    test_all_experiments()

    #    test_all_experiments()
#     with cProfile.Profile() as profile:
#         main()
#
# stats = pstats.Stats(profile)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.dump_stats("results.prof")
# stats.print_stats(20)





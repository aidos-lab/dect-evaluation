import torch
import os

"""
This is an almost copy of the main program and runs ALL experiments in the experiment 
folder for 1 epoch to check if everything works. 
It does not interfere with the configs.
"""

from main import run_experiment


def test_all_experiments():
    experiments = os.listdir("./experiment")
    for experiment in experiments:
        print("Running experiment", experiment)
        run_experiment(experiment, dev=True)


if __name__ == "__main__":
    test_all_experiments()

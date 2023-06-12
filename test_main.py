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
    experiments = [
            "modelnet_simple",
            ]
    experiments = [
            "gnnmnist_classification"
            "modelnet_points100_classification",
            "cnn_theta_sweep",
            "linear_theta_sweep",
            "modelnet_simple",
            "manifold_classification",
            "modelnet40_simple",
            ]
    experiments = [
            "gnnmnist_classification"
            ]
    for experiment in experiments: 
        print("Running experiment", experiment)
        run_experiment(experiment,dev=True)


if __name__ == "__main__":
    test_all_experiments()
    """ import cProfile """
    """ cProfile.run("test_all_experiments()","stats.prof",sort="cumtime") """
    """"""
    """ import pstats """
    """ stats = pstats.Stats("stats.prof") """
    """ stats.sort_stats("cumtime").print_stats(30) """



    #    test_all_experiments()
#     with cProfile.Profile() as profile:
#         main()
#
# stats = pstats.Stats(profile)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.dump_stats("results.prof")
# stats.print_stats(20)





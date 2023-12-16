import torch
from main import compute_avg

from main import Experiment
from logger import Logger, timing

mylogger = Logger()


def main():
    accs = []
    for _ in range(1):
        print("Running experiment", "ect_cnn_best.yaml")
        exp = Experiment(
            "./experiment/weighted_mnist/wect.yaml", logger=mylogger, dev=True
        )
        # exp = Experiment(
        #     "./experiment/manifold_classification/ect_cnn_faces.yaml",
        #     logger=mylogger,
        #     dev=True,
        # )
        loss, acc = exp.run()
        accs.append(acc)
    compute_avg(torch.tensor(accs))


if __name__ == "__main__":
    main()

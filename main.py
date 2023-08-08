import torch
from omegaconf import OmegaConf
from utils import count_parameters
from utils import listdir
from logger import Logger, timing
from metrics.metrics import compute_confusion, compute_acc

import loaders.factory as loader
import time

torch.cuda.empty_cache()

mylogger = Logger()


class Experiment:
    def __init__(self, experiment, logger, dev=True):
        """
        Creates the setup and does inits
        - Loads datamodules
        - Loads models
        - Initializes logger
        """

        self.config = OmegaConf.load(experiment)
        self.dev = dev
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger.log("Setup")
        self.logger.wandb_init(self.config.meta)

        # Load the dataset
        self.dm = loader.load_module("dataset", self.config.data)

        print(self.config.model)
        # Load the model
        model = loader.load_module("model", self.config.model)

        # Send model to device
        self.model = model.to(self.device)

        # Loss function and optimizer.
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.trainer.lr)

        # Log info
        self.logger.log(
            f"{self.config.model.module} has {count_parameters(self.model)} trainable parameters"
        )

    @timing(mylogger)
    def run(self):
        """
        Runs an experiment given the loaded config files.
        """
        start = time.time()
        for epoch in range(self.config.trainer.num_epochs):
            self.run_epoch()

            if epoch % 10 == 0:
                end = time.time()
                self.compute_metrics(epoch)
                self.logger.log(
                    msg=f"Training the model 10 epochs took: {end - start:.2f} seconds."
                )
                start = time.time()

        self.finalize_run()

    def run_epoch(self):
        for batch in self.dm.train_dataloader():
            batch_gpu, y_gpu = batch.to(self.device), batch.y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(batch_gpu)

            loss = self.loss_fn(pred, y_gpu)
            loss.backward()

            self.optimizer.step()

    @timing(mylogger)
    def finalize_run(self):
        # Compute accuracy
        loss, acc = compute_acc(self.model, self.dm.test_dataloader(), self.loss_fn)

        # Compute confusion
        cfm = compute_confusion(self.model, self.dm.test_dataloader())

        # Log statements
        self.logger.log(
            f"Test accuracy {acc:.2f},\n Confusion Matrix:\n {cfm}.",
            params={
                "thetas": self.config.model.num_thetas,
                "test_acc": acc,
                "test_loss": loss,
            },
        )

    def compute_metrics(self, epoch):
        loss, acc = compute_acc(self.model, self.dm.val_dataloader(), self.loss_fn)

        # Log statements to console
        self.logger.log(
            msg=f"epoch {epoch} | train loss {loss.item():.2f} | Accuracy {acc:.2f}",
            params={"epoch": epoch, "val_loss": loss.item(), "val_acc": acc},
        )


def main():
    experiment = "letter_high_classification"
    for experiment in listdir(f"./experiment/{experiment}"):
        print("Running experiment", experiment)
        exp = Experiment(experiment, logger=mylogger, dev=True)
        exp.run()


if __name__ == "__main__":
    main()

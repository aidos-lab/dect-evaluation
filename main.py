import torch
from omegaconf import OmegaConf
from utils import count_parameters
from utils import listdir
from logger import Logger, timing
from metrics.metrics import compute_confusion, compute_acc
import loaders.factory as loader
import time
<<<<<<< HEAD

import torchmetrics

torch.cuda.empty_cache()
mylogger = Logger()


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print("stopped early")
                return True
        return False


def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm**2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm


class Experiment:
    def __init__(self, experiment, logger, dev=True):
        self.config = OmegaConf.load(experiment)
        self.dev = dev
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger.log("Setup")
        self.logger.wandb_init(self.config.meta)

        # Load the dataset
        self.dm = loader.load_module("dataset", self.config.data)
        print(self.dm.entire_ds[0].x.shape)
        print(self.config.model)
        # Load the model
        model = loader.load_module("model", self.config.model)

        # Send model to device
        self.model = model.to(self.device)

        # Loss function and optimizer.
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [{"params": self.model.parameters()}],
            lr=self.config.trainer.lr,
            # weight_decay=1e-7,
            # eps=1e-4,
        )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.2, patience=10, verbose=True
        # )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[500], gamma=0.1
        )

        self.early_stopper = EarlyStopper()

        self.accuracy_list = []

        # Log info
        # Log info
        self.logger.log(f"Configurations:\n {OmegaConf.to_yaml(self.config)}")
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

            # if self.early_stopper(val_loss):
            #     break

            if epoch % 10 == 0:
                end = time.time()
                self.compute_metrics(epoch, end - start)
                start = time.time()

        # Compute test accuracy
        loss, acc, roc = compute_acc(
            self.model, self.dm.test_dataloader(), self.config.model.num_classes
        )
        self.accuracy_list.append(acc)

        # Log statements
        self.logger.log(
            f"Test accuracy: {acc:.3f}",
            params={
                # "thetas": self.config.model.num_thetas,
                "test_acc": acc,
                "test_loss": loss,
            },
        )
        return loss, acc

    def run_epoch(self):
        self.model.train()
        for batch in self.dm.train_dataloader():
            batch_gpu, y_gpu = batch.to(self.device), batch.y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch_gpu)
            loss = self.loss_fn(pred, y_gpu)
            loss.backward()
            clip_grad(self.model, 5)
            self.optimizer.step()
            # raise "hello"

        val_loss, _, _ = compute_acc(
            self.model, self.dm.val_dataloader(), self.config.model.num_classes
        )
        self.scheduler.step()
        return val_loss

        # del batch_gpu, y_gpu, pred, loss

    def compute_metrics(self, epoch, run_time):
        val_loss, val_acc, val_roc = compute_acc(
            self.model, self.dm.val_dataloader(), self.config.model.num_classes
        )
        train_loss, train_acc, _ = compute_acc(
            self.model, self.dm.train_dataloader(), self.config.model.num_classes
        )

        # Log statements to console
        self.logger.log(
            msg=f"epoch {epoch} | Train Loss {train_loss.item():.3f} | Val Loss {val_loss.item():.3f} | Train Accuracy {train_acc:.3f} | Val Accuracy {val_acc:.3f} | Run time {run_time:.2f} ",
            params={"epoch": epoch, "val_acc": val_acc},
        )


def compute_avg(acc: torch.Tensor):
    # torch.save(self.model.ectlayer.v, "test.pt")
    final_acc_mean = torch.mean(acc)
    final_acc_std = torch.std(acc)
    print(acc)
    # Log statements
    mylogger.log(
        f"Final accuracy {final_acc_mean:.4f} with std {final_acc_std:.4f}.",
    )


def main():
    experiments = [
        # "DD",
        # "ENZYMES",
        # "IMDB-BINARY",
        "Letter-high",
        # "Letter-med",
        # "Letter-low",
        # "gnn_mnist_classification",
        # "gnn_cifar10_classification",
        # "PROTEINS_full",
        # "REDDIT-BINARY",
        # "OGB-MOLHIV"
        # "dhfr",
        # "bzr",
        # "cox2"
    ]

    for experiment in experiments:
        for config in listdir(f"./experiment/{experiment}"):
            accs = []
            for _ in range(5):
                print("Running experiment", experiment, config)
                exp = Experiment(config, logger=mylogger, dev=True)
                loss, acc = exp.run()
                accs.append(acc)
            compute_avg(torch.tensor(accs))


if __name__ == "__main__":
    main()
=======
import os

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import confusion_matrix


def compute_confusion(model, loader):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch_gpu, y_gpu = batch.to(device), batch.y.to(device)
            y_pred.append(model(batch_gpu))
            y_true.append(y_gpu)

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred).max(axis=1)[1]
        cfm = confusion_matrix(
            y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        )
    return cfm


def compute_acc(model, loader, loss_fn):
    correct = 0
    total = 0
    loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch_gpu, y_gpu = batch.to(device), batch.y.to(device)
            y_pred.append(model(batch_gpu))
            y_true.append(y_gpu)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        loss = torch.sqrt(loss_fn(y_pred, y_true))
        y_pred = y_pred.max(axis=1)[1]
        correct = (y_pred == y_true).float().sum()
        acc = correct / len(y_true)
    return loss, acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(config, run=None):
    dm = load_datamodule(name=config.data.name, config=config.data.config)
    """ dm.info() """
    model = load_model(name=config.model.name, config=config.model.config)
    log_msg(f"{config.model.name} has {count_parameters(model)} trainable parameters")
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.lr)

    start = time.time()
    loss = torch.empty(0)
    for epoch in range(config.trainer.num_epochs):
        for batch in dm.train_dataloader():
            batch_gpu, y_gpu = batch.to(device), batch.y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_gpu)
            loss = loss_fn(pred, y_gpu)
            loss.backward()
            optimizer.step()
        if run:
            run.log({"epoch": epoch, "train_loss": loss.item()})
        if epoch % 10 == 0:
            end = time.time()
            loss, acc = compute_acc(model, dm.val_dataloader(), loss_fn)
            log_msg(
                f"epoch {epoch} | train loss {loss.item():.2f} | Accuracy {acc:.2f} | time {start-end:.2f}"
            )
            if run:
                run.log({"epoch": epoch, "val_loss": loss.item()})
                run.log({"epoch": epoch, "val_acc": acc})
            start = time.time()

    loss, acc = compute_acc(model, dm.test_dataloader(), loss_fn)
    cfm = compute_confusion(model, dm.test_dataloader())
    log_msg(f"Test accuracy {acc:.2f}")
    print(cfm)
    if run:
        run.log({"thetas": config.model.config.num_thetas, "test_acc": acc})  # type: ignore
        run.log({"test_loss": loss})  # type: ignore
    return run


def run_experiment(experiment, dev=False):
    files = glob.glob(f"./experiment/{experiment}/*")

    for file in files:
        config = OmegaConf.load(file)
        log_msg(f"\n{OmegaConf.to_yaml(config, resolve=True)}")  # type: ignore

        tags = [config.model.name, config.data.name]
        if not dev:
            run = wandb.init(
                project="desct-test" if dev else "desct-final",
                name=experiment,
                tags=tags,
                reinit=True,
                config=OmegaConf.to_container(config, resolve=True),  # type: ignore
            )
            run = train_model(config, run)
            run.join()  # type: ignore
        else:
            run = train_model(config, run=None)


if __name__ == "__main__":
    experiments = os.listdir("./experiment")
    experiments = ["letter_high_classification"]
    for experiment in experiments:
        print("Running experiment", experiment)
        run_experiment(experiment, dev=True)
>>>>>>> main

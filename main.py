import wandb
from logs import log_msg
import glob
from datasets import load_datamodule
from models import load_model
from omegaconf import OmegaConf
import torch
import time
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
        cfm = confusion_matrix(y_true.cpu().detach().numpy(),y_pred.cpu().detach().numpy())
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


def train_model(config,run=None):
    dm = load_datamodule(
        name=config.data.name,
        config=config.data.config
    )
    """ dm.info() """
    model = load_model(
            name=config.model.name,
            config=config.model.config
            )
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
            log_msg(f"epoch {epoch} | train loss {loss.item():.2f} | Accuracy {acc:.2f} | time {start-end:.2f}")
            if run:
                run.log({"epoch": epoch, "val_loss":loss.item()})
                run.log({"epoch": epoch, "val_acc":acc})
            start = time.time()

    loss,acc = compute_acc(model,dm.test_dataloader(),loss_fn)
    cfm = compute_confusion(model,dm.test_dataloader())
    log_msg(f"Test accuracy {acc:.2f}")
    print(cfm)
    if run:
        run.log({"thetas": config.model.config.num_thetas, "test_acc": acc})  # type: ignore
        run.log({"test_loss": loss})  # type: ignore
    return run


def run_experiment(experiment,dev=False):
    files = glob.glob(f"./experiment/{experiment}/*")

    for file in files:
        config = OmegaConf.load(file)
        log_msg(f"\n{OmegaConf.to_yaml(config, resolve=True)}") # type: ignore

        tags = [
            config.model.name,
            config.data.name
                ]
        if not dev:
            run = wandb.init(
                    project="desct-test" if dev else "desct-final", 
                    name=experiment,
                    tags=tags,
                    reinit=True,
                    config=OmegaConf.to_container(config, resolve=True) # type: ignore
                    )
            run = train_model(config,run)
            run.join() # type: ignore
        else:
            run = train_model(config,run=None)

        

if __name__ == "__main__":
    experiments = os.listdir("./experiment")
    for experiment in experiments: 
        print("Running experiment", experiment)
        run_experiment(experiment)



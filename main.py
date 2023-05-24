import wandb
from logs import log_msg
import glob
from datasets import load_datamodule
from models import load_model
from omegaconf import OmegaConf
import torch
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_acc(model, loader, loss_fn):
    correct = 0
    total = 0
    loss = 0
    for batch in loader:
        with torch.no_grad():
            pred = model(batch.to(device))
            loss = torch.sqrt(loss_fn(pred, batch.y))
            correct += (pred.max(axis=1)[1] == batch.y).float().sum()
            total += batch.y.shape[0]
    acc = correct / total
    return loss, acc


def train_model(config):
    dm = load_datamodule(
        name=config.data.name,
        config=config.data.config
    )
    model = load_model(
            name=config.model.name,
            config=config.model.config
            )
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.lr)

    for epoch in range(config.trainer.num_epochs):
        for batch in dm.train_dataloader():
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = torch.sqrt(loss_fn(pred, batch.y))
            loss.backward()
            optimizer.step()
        wandb.log({"epoch": epoch, "train_loss": loss.item()})
        if epoch % 10 == 0:
            log_msg(f"epoch {epoch} | train loss {loss.item():.2f}")
            loss, acc = compute_acc(model, dm.val_dataloader(), loss_fn)
            log_msg(f"Accuracy {acc:.2f}")
            wandb.log({"epoch": epoch, "val_loss":loss.item()})
            wandb.log({"epoch": epoch, "val_acc":acc})

    loss,acc = compute_acc(model,dm.test_dataloader(),loss_fn)
    log_msg(f"Accuracy {acc:.2f}")
    wandb.log({"thetas": config.model.config.num_thetas, "test_acc": acc})  # type: ignore
    wandb.log({"test_loss": loss})  # type: ignore


def main():
    experiment = 'cnn_theta_sweep'
    files = glob.glob(f"./config/{experiment}/*")
    for file in files:
        config = OmegaConf.load(file)
        tags = [
            config.model.name,
            config.data.name
                ]
        wandb.init(project="desct", name=experiment,tags=tags)
        train_model(config)

if __name__ == "__main__":
    main()
#     with cProfile.Profile() as profile:
#         main()
#
# stats = pstats.Stats(profile)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.dump_stats("results.prof")
# stats.print_stats(20)





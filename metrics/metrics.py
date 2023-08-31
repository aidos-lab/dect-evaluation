import torch
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import AUROC


def compute_confusion(model, loader):
    y_true = []
    y_pred = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


import torchmetrics


def compute_acc(model, loader, num_classes=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    auroc = AUROC(task="multiclass", num_classes=num_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = torch.tensor([0.0], device=device)
    with torch.no_grad():
        for batch in loader:
            batch_gpu, y_gpu = batch.to(device), batch.y.to(device)
            logits = model(batch_gpu)
            loss += loss_fn(logits, y_gpu)
            auroc(logits, y_gpu)
            acc(logits, y_gpu)
    a = acc.compute()
    roc = auroc.compute()
    acc.reset()
    auroc.reset()
    return loss, a, roc

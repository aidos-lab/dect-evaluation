import torch
from sklearn.metrics import confusion_matrix


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
        cfm = confusion_matrix(y_true.cpu().detach().numpy(),y_pred.cpu().detach().numpy())
    return cfm



def compute_acc(model, loader, loss_fn):
    correct = 0
    loss = 0
    y_true = []
    y_pred = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


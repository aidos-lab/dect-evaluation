import argparse
from utils import Parser
from types import SimpleNamespace
import json
from models import CNN
from models import ToyModel, ECTModel
from datasets import CustomDataLoader
from pretty_simple_namespace import pprint
import torch

parser = Parser()
config = parser.parse()

model = ECTModel(config.model)
loader = CustomDataLoader(config.dataset)

# info = loader.info()
# batch = info.batch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# batch = batch.to(device)
# pred = model(batch)

# print(pred)
# print(pred.shape)
#

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Starting training...")
losses = []
for epoch in range(2000):
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        optimizer.step()
    if epoch % 10 == 1:
        print(f"Epoch {epoch} | Train Loss {loss}")
    
config.dataset.dataset_params.train=False
loader = CustomDataLoader(config.dataset)

correct = 0
total = 0
for batch in loader:
    with torch.no_grad():
        pred=model(batch.to(device))
        correct += (pred.max(axis=1)[1] == batch.y).float().sum()
        total += batch.y.shape[0]
acc = correct / total
print(f"Accuracy {acc}")








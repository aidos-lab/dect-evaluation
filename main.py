import logging
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
timestr = time.strftime("%Y%m%d-%H%M%S")
fh = logging.FileHandler(filename=f"./logs/test-{timestr}.logs",mode="a")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

from utils import Parser
import json
from models import ECTCNNModel
from models import ECTPointsLinearModel

from datasets import CustomDataLoader
import torch

parser = Parser()
config = parser.parse()
logger.info(config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = ECTPointsLinearModel(config.model)
loader = CustomDataLoader(config.dataset)

logger.info(f"Number of params:{count_parameters(model)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

logger.info("Starting training...")
losses = []
for epoch in range(500):
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        logger.info(f"Epoch {epoch} | Train Loss {loss}")
  
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
logger.info(f"Accuracy {acc}")




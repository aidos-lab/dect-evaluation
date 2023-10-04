from models.layers.layers import EctPointsLayer
from models.base_model import ECTModelConfig
from dataclasses import dataclass
import torch
import geotorch
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch.nn as nn

scale = 100
num_classes = 40

x = torch.vstack(
    [torch.sigmoid(i * (torch.linspace(-1, 1, 100))) for i in range(num_classes)]
)

# p = torch.log(x).detach().numpy()

# for i in range(num_classes):
#     plt.plot(p[i])
# plt.show()


y = torch.LongTensor([i for i in range(num_classes)])

model = nn.Sequential(nn.Linear(100, 100), nn.ELU(), nn.Linear(100, num_classes))


out = model(x)
print(out)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2000):
    optimizer.zero_grad()
    pred = model(x)
    l = loss(pred, y)
    l.backward()
    optimizer.step()
    print("loss", l.item())


_, pred = torch.softmax(model(x), dim=1).max(dim=1)

print(pred)

# im = out.detach().numpy()

# plt.scatter(im[:, 0], im[:, 1])
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.show()

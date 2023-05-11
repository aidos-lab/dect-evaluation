import argparse
from utils import Parser
from types import SimpleNamespace
import json
from models import CNN
from models import ToyModel
from datasets import CustomDataLoader
from pretty_simple_namespace import pprint
import torch



parser = Parser()
config = parser.parse()

config.model.num_thetas = 50
config.model.bump_steps = 30 # Sampling density in ect curve
config.model.num_features = 3
config.model.device = "cpu"
config.model.R = 3
config.model.scale = 200

model = ToyModel(config.model)
loader = CustomDataLoader(config.dataset)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = model.to(device)

print("Starting training...")
losses = []
for epoch in range(500):
    for batch in loader:
        batch = batch.to(device) 
        optimizer.zero_grad()
        pred = model(batch)
        loss = torch.sqrt(loss_fn(pred, batch.y))    
        loss.backward()
        optimizer.step()
        break
    break

    # if epoch % 10 == 0:
    #     with torch.no_grad():
    #         pred=model(batch)
    #         acc = (pred.max(axis=1)[1] == batch.y).float().sum() / batch.y.shape[0]
    #     print(f"Epoch {epoch} | Train Loss {loss}, Accuracy {acc}")
    #














# config.dataset.dataset_params.train=True
# train_loader = CustomDataLoader(config.dataset)
# config.dataset.dataset_params.train=False
# test_loader = CustomDataLoader(config.dataset)


#
# loaders = {"train":train_loader,"test":test_loader}
#
#
#
# cnn = CNN()
# loss_func = nn.CrossEntropyLoss()   
# from torch import optim
# optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
#
# from torch.autograd import Variable
# num_epochs = 10
# def train(num_epochs, cnn, loaders):
#     cnn.train()
#     # Train the model
#     total_step = len(loaders['train'])
#         
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(loaders['train']):
#             
#             # gives batch data, normalize x when iterate train_loader
#             b_x = Variable(images)   # batch x
#             b_y = Variable(labels)   # batch y
#             output = cnn(b_x)[0]               
#             loss = loss_func(output, b_y)
#             
#             # clear gradients for this training step   
#             optimizer.zero_grad()           
#             
#             # backpropagation, compute gradients 
#             loss.backward()    
#             # apply gradients             
#             optimizer.step()                
#             
#             if (i+1) % 100 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#
# train(num_epochs, cnn, loaders)
#






# for x in loader.dataloader:
#     print(x)
# #     break   


# for x in loader:
#     print(x)
#     break   


#  ╭──────────────────────────────────────────────────────────╮
#  │ NEW STUFF                                                │
#  ╰──────────────────────────────────────────────────────────╯

# from torch_topological.nn import CubicalComplex
# from torch_topological.nn import SummaryStatisticLoss
# from torch_topological.utils import SelectByDimension
# from torch_topological.data import sample_from_unit_cube
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torchvision
# from torch_geometric.utils import subgraph
# import torch_geometric
# from PIL import Image, ImageOps
# import scipy
# from torchvision.datasets import MNIST
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import networkx as nx
# from torch_geometric.datasets import GNNBenchmarkDataset
# import torch_geometric.transforms as tf
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import subgraph
# import matplotlib.pyplot as plt
# import networkx as nx
# import torch_geometric
# import torch
# import torch.nn as nn
# import geotorch
#
#
# # Proteins dataset in TUDataset 3d attributes
# # Letter dataset in TU dataset
# # _MD in the tudataset
#
#
# #  ╭──────────────────────────────────────────────────────────╮
# #  │ Define Model                                             │
# #  ╰──────────────────────────────────────────────────────────╯
#
# class GEctLayer(nn.Module):
#     """docstring for EctLayer."""
#     def __init__(self):
#         super(GEctLayer, self).__init__()
#         self.num_thetas = 50
#         self.bump_steps = 30 # Sampling density in ect curve
#         self.num_features = 3
#         #self.thetas = torch.nn.Parameter(torch.linspace(0,.2*torch.pi,self.num_thetas,dtype=torch.float), requires_grad=True)
#         self.v = torch.nn.Parameter(torch.rand(size=(self.num_thetas,self.num_features)))
#         self.R = 3
#         self.scale = 200
#         self.lin = torch.linspace(-self.R,self.R,self.bump_steps).view(-1,1,1).to(device)
#     def bump(self,pts,labels=None,ng=1):
#         ecc = torch.sigmoid(self.scale*(self.lin - pts[0,...])).to(device) - torch.sigmoid(self.scale*(self.lin - pts[1,...])).to(device)
#         if labels is None:
#             print(ecc.shape)
#             return ecc.sum(axis=1)
#         else:
#             out = torch.zeros((ecc.shape[0], ng, ecc.shape[2]), dtype=ecc.dtype).to(device)
#             return out.index_add_(1, labels, ecc).movedim(0,1)
#     def forward(self,data):
#         #v = self.v / torch.norm(self.v)
#         #v = torch.stack([torch.cos(self.thetas),torch.sin(self.thetas)])
#         nh = torch.tensor(data.x,dtype=torch.float)@self.v.T
#         node_pairs = torch.stack([nh,self.R*torch.ones(nh.shape).to(device)])
#         edge_pairs = torch.stack([nh[data.edge_index].max(dim=0)[0],self.R*torch.ones(data.edge_index.shape[1],self.num_thetas).to(device)])
#         ect = self.bump(node_pairs,data.batch,ng=data.num_graphs) - self.bump(edge_pairs,data.batch[data.edge_index[0,:]],ng=data.num_graphs) / 2
#         return ect.reshape(-1,1500)
#
# # bs = 128
# # gect = GEctLayer()
# # loader = DataLoader(dataset,batch_size=bs, shuffle=False)
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # gect = gect.to(device)
# # for idx,batch in enumerate(loader):
# #     batch = batch.to(device) 
# #     output = gect(batch)
# #     # print(output.shape, batch.y.shape)
#
#
# class ToyModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ectlayer = GEctLayer() 
#         geotorch.sphere(self.ectlayer,"v")
#         self.linear = torch.nn.Linear(1500, 10)
#     def forward(self, x):
#         x = self.ectlayer(x)
#         x = self.linear(x)
#         return x
#
# #  ╭──────────────────────────────────────────────────────────╮
# #  │ Data prep                                                │
# #  ╰──────────────────────────────────────────────────────────╯
#
# class ThresholdTransform(object):
#   # def __init__(self):
#   #     pass
#   def __call__(self, data):
#       #nodes = (data.x>0.3).nonzero()[:,0]
#     #s = subgraph(nodes,data.edge_index,relabel_nodes=True)
#     x = torch.hstack([data.pos,data.x]) 
#     x -= torch.tensor(0.5)
#     new_data = torch_geometric.data.Data(x=x, edge_index=data.edge_index,y=data.y)
#     return new_data
#
# dataset = GNNBenchmarkDataset(name="MNIST",root='./GNNBenchmarkdataset',pre_transform=ThresholdTransform())
#
# bs = 128
# gect = GEctLayer()
# loader = DataLoader(dataset,batch_size=bs, shuffle=False)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for idx,batch in enumerate(loader):
#     break
#
#
# #  ╭──────────────────────────────────────────────────────────╮
# #  │ Train Model                                              │
# #  ╰──────────────────────────────────────────────────────────╯
#
# NUM_GRAPHS_PER_BATCH=100
# model = ToyModel()
#
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
# loader = DataLoader(dataset,batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
#
# print("Starting training...")
# losses = []
# for epoch in range(500):
#     for batch in loader:
#         batch = batch.to(device)  
#         optimizer.zero_grad() 
#         pred = model(batch)
#         loss = torch.sqrt(loss_fn(pred, batch.y))     
#         loss.backward()
#         optimizer.step() 
#     losses.append(loss)
#     if epoch % 10 == 0:
#         with torch.no_grad():
#             pred=model(batch)
#             acc = (pred.max(axis=1)[1] == batch.y).float().sum() / batch.y.shape[0]
#         print(f"Epoch {epoch} | Train Loss {loss}, Accuracy {acc}")
#
#
# correct = 0
# total = 0
# for batch in loader:
#     with torch.no_grad():
#         pred=model(batch.to(device))
#         correct += (pred.max(axis=1)[1] == batch.y).float().sum() 
#         total += batch.y.shape[0]
# acc = correct / total 
# print(f"Accuracy {acc}")
#
#
#
# v = model.ectlayer.v.cpu().detach().numpy().T
# plt.clf()
# plt.scatter(v[0,:],v[1,:])
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.savefig(f"general_angle_epoch_{epoch}.jpg")
#
#
#
#
#
#
#
#
#
#
#
#
#

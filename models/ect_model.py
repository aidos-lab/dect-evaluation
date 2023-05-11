import torch
import geotorch
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch_geometric




#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯

class GEctLayer(nn.Module):
    """docstring for EctLayer."""
    def __init__(self,config = None):
        super(GEctLayer, self).__init__()
        self.config = config
        self.device = "cpu"
        if config:
            self.num_thetas = config.num_thetas
            self.bump_steps = config.bump_steps # Sampling density in ect curve
            self.num_features = config.num_features
            self.R = config.R
            self.scale = config.scale
        else:
            self.num_thetas = 50
            self.bump_steps = 30 # Sampling density in ect curve
            self.num_features = 3
            self.R = 3
            self.scale = 200

        self.v = torch.nn.Parameter(torch.rand(size=(self.num_thetas,self.num_features)))
        self.lin = torch.linspace(-self.R,self.R,self.bump_steps).view(-1,1,1).to(self.device)
    def bump(self,pts,labels=None,ng=1):
        ecc = torch.sigmoid(self.scale*(self.lin - pts[0,...])).to(self.device) - torch.sigmoid(self.scale*(self.lin - pts[1,...])).to(self.device)
        if labels is None:
            print(ecc.shape)
            return ecc.sum(axis=1)
        else:
            out = torch.zeros((ecc.shape[0], ng, ecc.shape[2]), dtype=ecc.dtype).to(self.device)
            return out.index_add_(1, labels, ecc).movedim(0,1)
    def forward(self,data):
        nh = data.x@self.v.T
        node_pairs = torch.stack([nh,self.R*torch.ones(nh.shape).to(self.device)])
        edge_pairs = torch.stack([nh[data.edge_index].max(dim=0)[0],self.R*torch.ones(data.edge_index.shape[1],self.num_thetas).to(self.device)])
        ect = self.bump(node_pairs,data.batch,ng=data.num_graphs) - self.bump(edge_pairs,data.batch[data.edge_index[0,:]],ng=data.num_graphs) / 2
        return ect.reshape(-1,self.num_thetas*self.bump_steps)


class ToyModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ectlayer = GEctLayer(config) 
        geotorch.sphere(self.ectlayer,"v")
        self.linear = torch.nn.Linear(1500, 10)
    def forward(self, x):
        x = self.ectlayer(x)
        x = self.linear(x)
        return x


import torch
import geotorch
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch_geometric
import functools
import operator

from pretty_simple_namespace import pprint

#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯

class GEctLayer(nn.Module):
    """docstring for EctLayer."""
    def __init__(self,config = None):
        super(GEctLayer, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config:
            self.num_thetas = config.num_thetas
            self.bump_steps = config.bump_steps # Sampling density in ect curve
            self.num_features = config.num_features
            self.R = config.R
            self.scale = config.scale
        self.v = torch.nn.Parameter(torch.rand(size=(self.num_thetas,self.num_features))-0.5)
        # self.v = torch.rand(size=(self.num_thetas,self.num_features))
        # self.v = torch.nn.functional.normalize(self.v,dim=0).to(self.device)
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
        #edge_pairs = torch.stack([nh[data.edge_index].max(dim=0)[0],self.R*torch.ones(data.edge_index.shape[1],self.num_thetas).to(self.device)])
        ect = self.bump(node_pairs,data.batch,ng=data.num_graphs) #- self.bump(edge_pairs,data.batch[data.edge_index[0,:]],ng=data.num_graphs) / 2
        return ect#.reshape(-1,self.num_thetas*self.bump_steps)


class GEctPointsLayer(nn.Module):
    """docstring for EctLayer."""
    def __init__(self,config = None):
        super(GEctPointsLayer, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config:
            self.num_thetas = config.num_thetas
            self.bump_steps = config.bump_steps # Sampling density in ect curve
            self.num_features = config.num_features
            self.R = config.R
            self.scale = config.scale
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
        ect = self.bump(node_pairs,data.batch,ng=data.num_graphs) 
        return ect.reshape(-1,self.num_thetas*self.bump_steps)

class ToyModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ectlayer = GEctLayer(config) 
        geotorch.sphere(self.ectlayer,"v")
        self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, 10)
    def forward(self, x):
        x = self.ectlayer(x)
        x = self.linear(x)
        return x

class ECTPointsModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        pprint(config)
        self.ectlayer = GEctLayer(config) 
        geotorch.sphere(self.ectlayer,"v")
        self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, config.output)
    def forward(self, x):
        x = self.ectlayer(x)
        x = torch.nn.functional.normalize(x,dim=1)
        x = self.linear(x) # print("linear",x.max())
        return x

class ECTPointsLinearModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        pprint(config)
        self.ectlayer = GEctPointsLayer(config) 
        geotorch.sphere(self.ectlayer,"v")
        self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, config.hidden)
        self.linear2 = torch.nn.Linear(config.hidden, config.output)
    def forward(self, x):
        x = self.ectlayer(x) / 100
        x = self.linear(x)
        x = self.linear2(x)
        return x



class ECTCNNModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        pprint(config)
        self.ectlayer = GEctLayer(config) 
        # geotorch.sphere(self.ectlayer,"v")
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
            # nn.Conv2d(16, 32, 5, 1, 2),     
            # nn.ReLU(),                      
            # nn.MaxPool2d(2),                
        )
        
        num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape))
        self.out = nn.Linear(num_features, 10)

    def forward(self, x):
        x = self.ectlayer(x) / 100
        x = x.unsqueeze(1)
        x = self.conv1(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output



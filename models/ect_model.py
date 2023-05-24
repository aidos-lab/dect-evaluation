import torch
import geotorch
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch_geometric
import functools
import operator
import pytorch_lightning as pl
from models.base_model import BaseModel

from models.layers.layers import GEctLayer, GEctPointsLayer

#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯

# class ToyModel(pl.LightningModule):
#     def __init__(self,config):
#         super().__init__()
#         self.ectlayer = GEctLayer(config) 
#         geotorch.sphere(self.ectlayer,"v")
#         self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, 10)
#     def forward(self, x):
#         x = self.ectlayer(x)
#         x = self.linear(x)
#         return x

# class ECTPointsModel(pl.LightningModule):
#     def __init__(self,config):
#         super().__init__()
#         self.ectlayer = GEctLayer(config) 
#         geotorch.sphere(self.ectlayer,"v")
#         self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, config.num_classes)
#     def forward(self, x):
#         x = self.ectlayer(x)
#         x = torch.nn.functional.normalize(x,dim=1)
#         x = self.linear(x) # print("linear",x.max())
#         return x
#


class ECTPointsLinearModel(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.ectlayer = GEctPointsLayer(config) 
        geotorch.sphere(self.ectlayer,"v")
        self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, config.hidden)
        self.linear2 = torch.nn.Linear(config.hidden, config.num_classes)
    def forward(self, x):
        x = self.ectlayer(x) / 100
        x = self.linear(x)
        x = self.linear2(x)
        return x

class ECTCNNModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.ectlayer = GEctLayer(config) 
        geotorch.sphere(self.ectlayer,"v")
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            #nn.MaxPool2d(kernel_size=2),    
            # nn.Conv2d(16, 32, 5, 1, 2),     
            # nn.ReLU(),                      
            # nn.MaxPool2d(2),                
        )
        num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape))
        self.out = nn.Linear(num_features, config.num_classes)

    def forward(self, x):
        x = self.ectlayer(x) / 100
        x = x.unsqueeze(1)
        x = self.conv1(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output



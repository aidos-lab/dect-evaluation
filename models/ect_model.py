import torch
import geotorch
import torch
import torch.nn as nn
import functools
import operator
from models.base_model import BaseModel

from models.layers.layers import GEctLayer, GEctPointsLayer

#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯


class ECTPointsLinearModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.ectlayer = GEctPointsLayer(config) 
        geotorch.constraints.sphere(self.ectlayer,"v")
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
        geotorch.constraints.sphere(self.ectlayer,"v")
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU()                      
        )
        num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape))
        self.out = nn.Linear(num_features, config.num_classes)

    def forward(self, x):
        x = self.ectlayer(x) / 100
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output



import torch
import geotorch
import torch
import torch.nn as nn
import functools
import operator
from models.base_model import BaseModel

from models.layers.layers import EctEdgesLayer, EctPointsLayer

#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯

#  ╭──────────────────────────────────────────────────────────╮
#  │ These two models only take the point cloud               │
#  │  structure into account.                                 │
#  ╰──────────────────────────────────────────────────────────╯

class ECTLinearPointsModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.ectlayer = EctPointsLayer(config)
        geotorch.constraints.sphere(self.ectlayer,"v")
        self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, config.hidden)
        self.linear2 = torch.nn.Linear(config.hidden, config.num_classes)

    def forward(self, batch):
        x = self.ectlayer(batch).reshape(-1,self.config.num_thetas*self.config.bump_steps)
        x = self.linear(x)
        x = self.linear2(x)
        return x


class ECTCNNPointsModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        """ self.ectlayer = torch.jit.script(GEctPointsLayer(config,self.v)) """
        self.ectlayer = EctPointsLayer(config)
        geotorch.constraints.sphere(self.ectlayer,"v")
        self.conv1 = nn.Sequential(         
        nn.Conv2d(1,16, kernel_size=3),
        nn.MaxPool2d(2)
        )
        num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape))
        self.linear1 = nn.Linear(num_features, config.num_classes)
    
    def forward(self,batch):
        x = self.ectlayer(batch) 
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)       
        x = self.linear1(x)
        return x


#  ╭──────────────────────────────────────────────────────────╮
#  │ The below two models take the                            │
#  │ points and edges into account.                           │
#  ╰──────────────────────────────────────────────────────────╯

class ECTCNNEdgesModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.ectlayer = EctEdgesLayer(config) 
        geotorch.constraints.sphere(self.ectlayer,"v")
        self.conv1 = nn.Sequential(         
        nn.Conv2d(1,16, kernel_size=3),
        nn.MaxPool2d(2)
        )
        num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape))
        self.out = nn.Linear(num_features, config.num_classes)

    def forward(self, x):
        x = self.ectlayer(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output




class ECTLinearEdgesModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.ectlayer = EctEdgesLayer(config) 
        geotorch.constraints.sphere(self.ectlayer,"v")
        self.linear = torch.nn.Linear(config.num_thetas*config.bump_steps, config.hidden)
        self.linear2 = torch.nn.Linear(config.hidden, config.num_classes)


    def forward(self, x):
        x = self.ectlayer(x).reshape(-1,self.config.num_thetas*self.config.bump_steps)
        x = self.linear(x)
        x = self.linear2(x)
        return x






""" nn.Conv2d(1, 8, kernel_size=3), """
""" nn.MaxPool2d(2), """
""" nn.Conv2d(8,16, kernel_size=3), """
""" nn.MaxPool2d(2), """
""" nn.Conv2d(16,32, kernel_size=3), """
""" nn.MaxPool2d(2) """

""" nn.Conv2d( """
"""     in_channels=1,               """
"""     out_channels=16,             """
"""     kernel_size=3,               """
"""     stride=1,                    """
"""     padding=1,                   """
""" ),                               """
""" nn.ReLU()                       """

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
        self.linear1 = torch.nn.Linear(config.num_thetas*config.bump_steps, config.hidden)
        self.linear2 = torch.nn.Linear(config.hidden,100)
        self.linear3 = nn.Linear(100, config.num_classes)

    def forward(self, batch):
        x = self.ectlayer(batch).reshape(-1,self.config.num_thetas*self.config.bump_steps)
        """ x = x / x.max(dim=1)[0].unsqueeze(1) """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x


class ECTCNNPointsModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.ectlayer = EctPointsLayer(config)
        geotorch.constraints.sphere(self.ectlayer,"v")
        self.conv1 = nn.Sequential(         
            nn.Conv2d(1,8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16, kernel_size=3),
            nn.MaxPool2d(2),
        )
        num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape))
        self.linear1 = nn.Linear(num_features, config.hidden)
        self.linear2 = nn.Linear(config.hidden, 100)
        self.linear3 = nn.Linear(100, config.num_classes)
    
    def forward(self,batch):
        x = self.ectlayer(batch) 
        x = x.unsqueeze(1)
        """ x = x / torch.max(torch.abs(x)) """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)       
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        return x




""" class ECTCNNPointsModel(BaseModel): """
"""     def __init__(self,config): """
"""         super().__init__(config) """
"""         self.ectlayer = EctPointsLayer(config) """
"""         geotorch.constraints.sphere(self.ectlayer,"v") """
"""         self.conv1 = nn.Sequential(          """
"""             nn.Conv2d(1,8, kernel_size=3), """
"""             nn.MaxPool2d(2), """
"""             nn.Conv2d(8,16, kernel_size=3), """
"""             nn.MaxPool2d(2), """
"""             nn.Conv2d(16,32, kernel_size=3), """
"""             nn.MaxPool2d(2), """
"""         ) """
"""         num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape)) """
"""         self.linear1 = nn.Linear(num_features, config.hidden) """
"""         self.linear2 = nn.Linear(config.hidden, 200) """
"""         self.linear3 = nn.Linear(200, config.num_classes) """
"""      """
"""     def forward(self,batch): """
"""         x = self.ectlayer(batch)  """
"""         x = x.unsqueeze(1) """
"""         x = self.conv1(x) """
"""         x = x.view(x.size(0), -1)        """
"""         x = self.linear1(x) """
"""         # Add Relu here """
"""         x = nn.functional.relu(x)                       """
"""         x = self.linear2(x) """
"""         x = nn.functional.relu(x)                       """
"""         x = self.linear3(x) """
"""         return x """
""""""

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
            nn.Conv2d(1,8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16, kernel_size=3),
            nn.MaxPool2d(2),
        )
        num_features = functools.reduce(operator.mul, list(self.conv1(torch.rand(1, config.bump_steps,config.num_thetas)).shape))
        self.linear1 = nn.Linear(num_features, config.hidden)
        self.linear2 = nn.Linear(config.hidden, 100)
        self.linear3 = nn.Linear(100, config.num_classes)

    def forward(self, x):
        x = self.ectlayer(x)
        x = x.unsqueeze(1)
        """ x = x + torch.amin(x,dim=(1,2,3)).view(-1,1,1,1) """
        """ x = x + torch.amax(x,dim=(1,2,3)).view(-1,1,1,1) """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)       
        x = self.linear1(x)
        x = nn.functional.relu(x)                      
        x = self.linear2(x)
        x = nn.functional.relu(x)                      
        x = self.linear3(x)
        return x




class ECTLinearEdgesModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.ectlayer = EctEdgesLayer(config) 
        geotorch.constraints.sphere(self.ectlayer,"v")
        self.linear1 = torch.nn.Linear(config.num_thetas*config.bump_steps, config.hidden)
        self.linear2 = torch.nn.Linear(config.hidden, 100)
        self.linear3 = nn.Linear(100, config.num_classes)


    def forward(self, batch):
        x = self.ectlayer(batch).reshape(-1,self.config.num_thetas*self.config.bump_steps)
        """ x = x / torch.abs(x).max(dim=1)[0].unsqueeze(1) """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
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

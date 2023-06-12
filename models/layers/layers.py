import torch
import torch.nn as nn
from torch_geometric.data import Batch
import sys


#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯


@torch.jit.script
def rel(nh,batch,out,lin):
    ecc = torch.nn.functional.relu(torch.sub(lin, nh)) 
    return torch.index_add(out,1, batch, ecc).movedim(0,1)



class EctPointsLayer(nn.Module):
    """docstring for EctLayer."""
    def __init__(self,config):
        super(EctPointsLayer, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.v = torch.nn.Parameter(torch.rand(size=(config.num_thetas,config.num_features))-0.5)#.to(device)
        self.num_thetas = config.num_thetas
        self.bump_steps = config.bump_steps # Sampling density in ect curve
        self.num_features = config.num_features
        self.batch_size = config.batch_size
        self.R = config.R
        self.scale = torch.tensor(config.scale,device=self.device)
        self.lin = torch.linspace(-self.R,self.R,self.bump_steps).view(-1,1,1).to(self.device)
        #self.out = torch.zeros(self.num_thetas, self.batch_size, self.bump_steps, dtype=torch.float32, device=self.device)
        """ self.zero_tensor_1dim=torch.tensor(0,dtype=torch.float32) """
   
    def forward(self,data):
        nh = (data.x@self.v.T).unsqueeze(0)
        out = torch.zeros(self.bump_steps, data.batch.max()+1, self.num_thetas, dtype=torch.float32, device=self.device)
        return rel(nh,data.batch,out,self.lin) 
    
    def extra_repr(self):
        print(vars(self.config))
        return ", ".join([f"{str(key)}={str(value)}" for key,value in vars(self.config).items()])


class EctEdgesLayer(EctPointsLayer):
    """docstring for EctLayer."""
    def __init__(self,config):
        super(EctEdgesLayer, self).__init__(config)
   
    def forward(self,data):
        nh = (data.x@self.v.T)
        eh,_ = nh[data.edge_index].max(dim=0)
        out = torch.zeros(self.bump_steps, data.batch.max()+1, self.num_thetas, dtype=torch.float32, device=self.device)
        return rel(nh,data.batch,out,self.lin) - rel(eh,data.batch[data.edge_index[0]],out,self.lin) 


class EctFacesLayer(EctPointsLayer):
    """docstring for EctLayer."""
    def __init__(self,config):
        super(EctFacesLayer, self).__init__(config)
   
    def forward(self,data):
        nh = (data.x@self.v.T).unsqueeze(0)
        eh,_ = nh[data.edge_index].max(dim=0)
        fh,_ = nh[data.face_index].max(dim=0)
        out = torch.zeros(self.bump_steps, data.batch.max()+1, self.num_thetas, dtype=torch.float32, device=self.device)
        return rel(nh,data.batch,out,self.lin) - rel(eh,data.batch[data.edge_index[0]],out,self.lin) + rel(fh,data.batch[data.face_index[0]],out,self.lin) 



#  ╭──────────────────────────────────────────────────────────╮
#  │ Below is backup just in case and to remember what i      │
#  │  did earlier. Do not use in exp                          │
#  ╰──────────────────────────────────────────────────────────╯


class GEctPointsBULayer(nn.Module):
    """docstring for EctLayer."""
    def __init__(self,config):
        super(GEctPointsBULayer, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_thetas = config.num_thetas
        self.bump_steps = config.bump_steps # Sampling density in ect curve
        self.num_features = config.num_features
        self.R = config.R
        self.scale = torch.tensor(config.scale,device=self.device)
        self.lin = torch.linspace(-self.R,self.R,self.bump_steps).view(-1,1,1).to(self.device)
        self.v = torch.nn.Parameter(torch.rand(size=(self.num_thetas,self.num_features),device=self.device)-0.5)

    def bump(self,pts,labels,ng=1):
        ecc = torch.sigmoid(self.scale*(self.lin - pts[0,...])) - torch.sigmoid(self.scale*(self.lin - pts[1,...]))
        out = torch.zeros((ecc.shape[0], ng, ecc.shape[2]), dtype=ecc.dtype,device=self.device)
        return out.index_add_(1, labels, ecc).movedim(0,1)

    def forward(self,data):
        nh = data.x@self.v.T
        node_pairs = torch.nn.functional.pad(nh.unsqueeze(0),(0,0,0,0,0,1),value=self.R)
        ect = self.bump(node_pairs,data.batch,ng=data.num_graphs) 
        return ect



class GEctPointsEdgesBULayer(nn.Module):
    """docstring for EctLayer."""
    def __init__(self,config):
        super(GEctPointsEdgesBULayer, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_thetas = config.num_thetas
        self.bump_steps = config.bump_steps # Sampling density in ect curve
        self.num_features = config.num_features
        self.R = config.R
        self.scale = torch.tensor(config.scale,device=self.device)
        self.lin = torch.linspace(-self.R,self.R,self.bump_steps).view(-1,1,1).to(self.device)
        self.v = torch.nn.Parameter(torch.rand(size=(self.num_thetas,self.num_features),device=self.device)-0.5)

    def bump(self,pts,labels,ng=1):
        ecc = torch.sigmoid(self.scale*(self.lin - pts[0,...])) - torch.sigmoid(self.scale*(self.lin - pts[1,...]))
        out = torch.zeros((ecc.shape[0], ng, ecc.shape[2]), dtype=ecc.dtype,device=self.device)
        return out.index_add_(1, labels, ecc).movedim(0,1)

    def forward(self,data):
        nh = data.x@self.v.T
        node_pairs = torch.nn.functional.pad(nh.unsqueeze(0),(0,0,0,0,0,1),value=self.R)
        edge_pairs = torch.nn.functional.pad(nh[data.edge_index].max(dim=0)[0].unsqueeze(0),(0,0,0,0,0,1),value=self.R)
        ect = self.bump(node_pairs,data.batch,ng=data.num_graphs) - self.bump(edge_pairs,data.batch[data.edge_index[0,:]],ng=data.num_graphs) / 2
        return ect
    
    def extra_repr(self):
        print(vars(self.config))
        return ", ".join([f"{str(key)}={str(value)}" for key,value in vars(self.config).items()])




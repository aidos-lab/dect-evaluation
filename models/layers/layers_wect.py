import torch
import torch.nn as nn
from torch_scatter import segment_coo
import geotorch
from models.config import EctConfig
from torch_geometric.data import Data

from typing import Protocol
from dataclasses import dataclass


def compute_wecc(nh, index, lin, weight,out):
    ecc = torch.nn.functional.sigmoid(500 * torch.sub(lin, nh)) * weight.view(1,-1,1)
    res = torch.index_add(out,1, index, ecc).movedim(0, 1)
    return res


def compute_wect_points(data, v, lin, out):
    # Compute the weights 
    edge_weights,_ = data.node_weights[data.edge_index].max(axis=0)
    face_weights,_ = data.node_weights[data.face].max(axis=0)

    nh = data.x @ v
    eh, _ = nh[data.edge_index].min(dim=0)
    fh, _ = nh[data.face].min(dim=0)
    return (
        compute_wecc(nh, data.batch, lin, data.node_weights,out)
    )

def compute_wect_edges(data, v, lin, out):
    # Compute the weights 
    edge_weights,_ = data.node_weights[data.edge_index].max(axis=0)
    face_weights,_ = data.node_weights[data.face].max(axis=0)
    nh = data.x @ v
    eh, _ = nh[data.edge_index].min(dim=0)
    fh, _ = nh[data.face].min(dim=0)
    return (
        compute_wecc(nh, data.batch, lin, data.node_weights,out)
        - compute_wecc(eh, data.batch[data.edge_index[0]], lin, edge_weights,out)
    )

def compute_wect_faces(data, v, lin, out):
    # Compute the weights 
    edge_weights,_ = data.node_weights[data.edge_index].max(axis=0)
    face_weights,_ = data.node_weights[data.face].max(axis=0)
    nh = data.x @ v
    eh, _ = nh[data.edge_index].min(dim=0)
    fh, _ = nh[data.face].min(dim=0)
    return (
        compute_wecc(nh, data.batch, lin, data.node_weights,out)
        - compute_wecc(eh, data.batch[data.edge_index[0]], lin, edge_weights,out)
        + compute_wecc(fh, data.batch[data.face[0]], lin, face_weights,out)
    )


class WectLayer(nn.Module):
    def __init__(self, config: EctConfig, fixed=False):
        super().__init__()
        self.config = config
        self.fixed = fixed
        self.lin = (
            torch.linspace(-config.R, config.R, config.bump_steps)
            .view(-1, 1, 1)
            .to(config.device)
        )
        if self.fixed:
            self.v = torch.vstack(
                [
                    torch.sin(torch.linspace(0, 2 * torch.pi, 256)),
                    torch.cos(torch.linspace(0, 2 * torch.pi, 256)),
                ]
            ).to(config.device)
        else:
            self.v = torch.nn.Parameter(
                torch.rand(size=(config.num_features, config.num_thetas)) - 0.5
            ).to(config.device)

        if config.ecc_type == "points":
            self.compute_wect = compute_wect_points
        elif config.ecc_type == "edges":
            self.compute_wect = compute_wect_edges
        elif config.ecc_type == "faces":
            self.compute_wect = compute_wect_faces

    def __post_init__(self):
        if self.fixed:
            geotorch.constraints.sphere(self, "v")

    def forward(self, data):
        out = torch.zeros(
            self.config.bump_steps,
            data.batch.max() + 1,
            self.config.num_thetas,
            dtype=torch.float32,
            device=self.config.device,
        )
        return self.compute_wect(data, self.v, self.lin, out)





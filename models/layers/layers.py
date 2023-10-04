import torch
import torch.nn as nn
from torch_scatter import segment_coo
import geotorch
from models.config import EctConfig


def compute_ecc(nh, index, lin, dim_size):
    ecc = torch.nn.functional.sigmoid(200 * torch.sub(lin, nh))
    # print(segment_coo(ecc, index.view(1, -1), reduce="sum").movedim(0, 1).shape)
    return segment_coo(ecc, index.view(1, -1), dim_size=dim_size, reduce="sum").movedim(
        0, 1
    )


def compute_ect_points(data, v, lin):
    nh = data.x @ v
    return compute_ecc(nh, data.batch, lin, dim_size=data.num_graphs)


def compute_ect_edges(data, v, lin):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    return compute_ecc(nh, data.batch, lin, dim_size=data.num_graphs) - compute_ecc(
        eh, data.batch[data.edge_index[0]], lin, dim_size=data.num_graphs
    )


def compute_ect_faces(data, v, lin):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    fh, _ = nh[data.face].max(dim=0)
    return (
        compute_ecc(nh, data.batch, lin, dim_size=data.num_graphs)
        - compute_ecc(eh, data.batch[data.edge_index[0]], lin, dim_size=data.num_graphs)
        + compute_ecc(fh, data.batch[data.face[0]], lin, dim_size=data.num_graphs)
    )


class EctLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config: EctConfig, fixed=False):
        super().__init__()
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
            self.compute_ect = compute_ect_points
        elif config.ecc_type == "edges":
            self.compute_ect = compute_ect_edges
        elif config.ecc_type == "faces":
            self.compute_ect = compute_ect_faces

    def __post_init__(self):
        if self.fixed:
            geotorch.constraints.sphere(self, "v")

    def forward(self, data):
        return self.compute_ect(data, self.v, self.lin)

import torch
import torch.nn as nn
from torch_scatter import segment_coo
import geotorch


def compute_ecc(nh, index, lin):
    ecc = torch.nn.functional.sigmoid(200 * torch.sub(lin, nh))
    return segment_coo(ecc, index.view(1, -1), reduce="sum").movedim(0, 1)


def compute_ect_points(data, v, lin):
    nh = data.x @ v
    return compute_ecc(nh, data.batch, lin)


def compute_ect_edges(data, v, lin):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    return (
        compute_ecc(nh, data.batch, lin)
        - compute_ecc(eh, data.batch[data.edge_index[0]], lin) / 2
    )


def compute_ect_faces(data, v, lin):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    fh, _ = nh[data.face].max(dim=0)
    return (
        compute_ecc(nh, data.batch, lin)
        - compute_ecc(eh, data.batch[data.edge_index[0]], lin) / 2
        + compute_ecc(fh, data.batch[data.face[0]], lin)
    )


class EctLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config, ecc_type="points", fixed=False):
        super().__init__()
        self.config = config
        self.lin = (
            torch.linspace(-self.config.R, self.config.R, self.config.bump_steps)
            .view(-1, 1, 1)
            .to(config.device)
        )
        if fixed:
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
            geotorch.constraints.sphere(self, "v")

        if ecc_type == "points":
            self.compute_ect = compute_ect_points
        elif ecc_type == "edges":
            self.compute_ect = compute_ect_edges
        elif ecc_type == "faces":
            self.compute_ect = compute_ect_faces

    def forward(self, data):
        return self.compute_ect(data, self.v, self.lin)

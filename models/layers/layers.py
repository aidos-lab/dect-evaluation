import torch
import torch.nn as nn
from torch_geometric.data import Batch
import sys


#  ╭──────────────────────────────────────────────────────────╮
#  │ Define Model                                             │
#  ╰──────────────────────────────────────────────────────────╯


@torch.jit.script
def rel(nh, batch, out, lin):
    ecc = torch.nn.functional.sigmoid(200 * torch.sub(lin, nh))
    return torch.index_add(out, 1, batch, ecc).movedim(0, 1)


class EctNodeHeightLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config):
        super(EctNodeHeightLayer, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.v = torch.nn.Parameter(
            torch.rand(size=(config.num_thetas, config.num_features)) - 0.5
        )
        self.num_thetas = config.num_thetas
        self.bump_steps = config.bump_steps  # Sampling density in ect curve
        self.num_features = config.num_features
        self.R = config.R
        self.lin = (
            torch.linspace(-self.R, self.R, self.bump_steps)
            .view(-1, 1, 1)
            .to(self.device)
        )

        # self.out = torch.zeros(self.num_thetas, self.batch_size, self.bump_steps, dtype=torch.float32, device=self.device)
        """ self.zero_tensor_1dim=torch.tensor(0,dtype=torch.float32) """

    def forward(self, data):
        nh = data.x @ self.v.T  # Removed unsqueese statement
        # out = torch.zeros(
        #     self.bump_steps,
        #     data.batch.max() + 1,
        #     self.num_thetas,
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        # print("batch shape", data.batch.shape)
        # print("node height", nh.shape)
        # print("lin shape", self.lin.shape)
        # print("out shape", out)
        return nh  # rel(nh, data.batch, out, self.lin)

    def extra_repr(self):
        print(vars(self.config))
        return ", ".join(
            [f"{str(key)}={str(value)}" for key, value in vars(self.config).items()]
        )


class Ect2DPointsLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config):
        super(Ect2DPointsLayer, self).__init__()
        self.config = config
        self.device = (
            "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.v = torch.nn.Parameter(
            torch.vstack(
                [
                    torch.sin(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                    torch.cos(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                ]
            ).T
        )
        self.num_thetas = config.num_thetas
        self.bump_steps = config.bump_steps  # Sampling density in ect curve
        self.num_features = config.num_features
        self.R = config.R
        self.lin = (
            torch.linspace(-self.R, self.R, self.bump_steps)
            .view(-1, 1, 1)
            .to(self.device)
        )

        # self.out = torch.zeros(self.num_thetas, self.batch_size, self.bump_steps, dtype=torch.float32, device=self.device)
        """ self.zero_tensor_1dim=torch.tensor(0,dtype=torch.float32) """

    def forward(self, data):
        nh = data.x @ self.v.T  # Removed unsqueese statement
        out = torch.zeros(
            self.bump_steps,
            data.batch.max() + 1,
            self.num_thetas,
            dtype=torch.float32,
            device=self.device,
        )
        # print("batch shape", data.batch.shape)
        # print("node height", nh.shape)
        # print("lin shape", self.lin.shape)
        # print("out shape", out)
        return rel(nh, data.batch, out, self.lin)

    def extra_repr(self):
        print(vars(self.config))
        return ", ".join(
            [f"{str(key)}={str(value)}" for key, value in vars(self.config).items()]
        )


class EctPointsLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config):
        super(EctPointsLayer, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.v = torch.nn.Parameter(
            torch.rand(size=(config.num_thetas, config.num_features)) - 0.5
        ).to(device)
        self.num_thetas = config.num_thetas
        self.bump_steps = config.bump_steps  # Sampling density in ect curve
        self.num_features = config.num_features
        self.R = config.R
        self.lin = (
            torch.linspace(-self.R, self.R, self.bump_steps)
            .view(-1, 1, 1)
            .to(self.device)
        )

    def forward(self, data):
        nh = data.x @ self.v.T  # Removed unsqueese statement
        out = torch.zeros(
            self.bump_steps,
            data.batch.max() + 1,
            self.num_thetas,
            dtype=torch.float32,
            device=self.device,
        )
        return rel(nh, data.batch, out, self.lin)

    def extra_repr(self):
        print(vars(self.config))
        return ", ".join(
            [f"{str(key)}={str(value)}" for key, value in vars(self.config).items()]
        )


class EctEdgesLayer(EctPointsLayer):
    """docstring for EctLayer."""

    def __init__(self, config):
        super(EctEdgesLayer, self).__init__(config)

    def forward(self, data):
        nh = data.x @ self.v.T
        eh, _ = nh[data.edge_index].max(dim=0)
        out = torch.zeros(
            self.bump_steps,
            data.batch.max() + 1,
            self.num_thetas,
            dtype=torch.float32,
            device=self.device,
        )
        return (
            rel(nh, data.batch, out, self.lin)
            - rel(eh, data.batch[data.edge_index[0]], out, self.lin) / 2
        )


class EctFacesLayer(EctPointsLayer):
    """docstring for EctLayer."""

    def __init__(self, config):
        super(EctFacesLayer, self).__init__(config)

    def forward(self, data):
        nh = data.x @ self.v.T
        eh, _ = nh[data.edge_index].max(dim=0)
        fh, _ = nh[data.face].max(dim=0)
        out = torch.zeros(
            self.bump_steps,
            data.batch.max() + 1,
            self.num_thetas,
            dtype=torch.float32,
            device=self.device,
        )
        return (
            rel(nh, data.batch, out, self.lin)
            - rel(eh, data.batch[data.edge_index[0]], out, self.lin) / 2
            + rel(fh, data.batch[data.face[0]], out, self.lin)
        )

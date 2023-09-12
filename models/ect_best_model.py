import geotorch
import torch
import torch.nn as nn
import functools
import operator
from models.base_model import BaseModel
from models.layers.layers import Ect2DPointsLayer, EctPointsLayer, EctNodeHeightLayer
from loaders.factory import register
from models.deepsets import PermEqui2_mean


class TestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ectlayer = EctPointsLayer(config)
        self.x_dim = config.bump_steps
        self.d_dim = 256
        self.phi = nn.Sequential(
            nn.Linear(self.x_dim, self.d_dim),
            nn.ELU(inplace=True),
            # nn.Linear(self.d_dim, self.d_dim),
            # nn.ELU(inplace=True),
            # nn.Linear(self.d_dim, self.d_dim),
            # nn.ELU(inplace=True),
        )
        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, self.d_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, 40),
        )

    def forward(self, batch):
        x = self.ectlayer(batch)
        x /= 50
        x -= 1
        x, _ = self.phi(x).max(1)
        # print(x.shape)
        # raise "hello"
        x = self.ro(x)
        return x


class ECTConvModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ectlayer = EctPointsLayer(config)
        self.x_dim = 10
        self.d_dim = 10
        self.phi = nn.Sequential(
            PermEqui2_mean(self.x_dim, self.d_dim),
            nn.ELU(inplace=True),
            PermEqui2_mean(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
            PermEqui2_mean(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(16320, config.hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(config.hidden, config.hidden // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(config.hidden // 2, config.num_classes),
        )

    def forward(self, batch):
        x = self.ectlayer(batch).unsqueeze(1)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        raise "hello"
        x = self.conv3(x)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x


class DeepsetsModel(BaseModel):
    def __init__(self, config):
        self.x_dim = 3
        self.d_dim = 128
        super().__init__(config)

        self.phi = nn.Sequential(
            PermEqui2_mean(self.x_dim, self.d_dim),
            nn.ELU(inplace=True),
            PermEqui2_mean(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
            PermEqui2_mean(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
        )
        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, self.d_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, 40),
        )
        print(self)

    def forward(self, data):
        phi_output = self.phi(data.x.view(-1, 100, 3))
        sum_output, _ = phi_output.max(1)
        ro_output = self.ro(sum_output)
        return ro_output


class ECTDeepsetsModel(BaseModel):
    def __init__(self, config):
        self.x_dim = config.num_thetas
        self.d_dim = 128
        super().__init__(config)
        self.ectlayer = EctNodeHeightLayer(config)
        geotorch.constraints.sphere(self.ectlayer, "v")
        self.phi = nn.Sequential(
            PermEqui2_mean(self.x_dim, self.d_dim),
            nn.ELU(inplace=True),
            PermEqui2_mean(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
            PermEqui2_mean(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
        )
        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, self.d_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, 40),
        )
        print(self)

    def forward(self, batch):
        x = self.ectlayer(batch).view(-1, 100, self.config.num_thetas)
        phi_output = self.phi(x)
        sum_output, _ = phi_output.max(1)
        ro_output = self.ro(sum_output)
        return ro_output


# class ECTCNNPointsModel(BaseModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.ectlayer = EctPointsLayer(config)
#         geotorch.constraints.sphere(self.ectlayer, "v")

#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(1, 32, kernel_size=(32, 1), stride=1, padding=0),
#         #     # nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #     # nn.MaxPool2d(kernel_size=(2), stride=2),
#         # )
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(32, 64, kernel_size=(5), stride=1, padding=0),
#         #     nn.BatchNorm2d(16),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=(2), stride=2),
#         # )
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(16, 32, kernel_size=(5), stride=1, padding=0),
#         #     nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=(2), stride=2),
#         # )
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
#         #     nn.BatchNorm2d(96),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=3, stride=2),
#         # )
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=3, stride=2),
#         # )
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
#         #     nn.BatchNorm2d(384),
#         #     nn.ReLU(),
#         # )

#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(1, 8, kernel_size=3),
#         #     nn.MaxPool2d(2),
#         #     nn.ReLU(),
#         #     nn.Conv2d(8, 16, kernel_size=3),
#         #     nn.MaxPool2d(2),
#         #     nn.ReLU(),
#         # )
#         # num_features = functools.reduce(
#         #     operator.mul,
#         #     list(
#         #         # self.conv3(
#         #         # self.conv2(
#         #         self.conv1(torch.rand(1, 1, config.bump_steps, config.num_thetas)).shape
#         #     ),
#         # )
#         # self.layernorm = nn.LayerNorm([self.config.num_thetas, self.config.bump_steps])
#         self.linear = nn.Sequential(
#             nn.Linear(self.config.num_thetas * self.config.bump_steps, config.hidden),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(config.hidden, config.hidden),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(config.hidden, config.num_classes),
#         )

#     def forward(self, batch):
#         x = self.ectlayer(batch)  # .unsqueeze(1)
#         # x = self.layernorm(x)
#         x /= 100
#         x -= 0.5
#         # x = self.conv1(x)
#         # x = self.conv2(x)
#         # x = self.conv3(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.linear(x)
#         return x


def initialize():
    register("model", TestModel)

from datasets.base_dataset import DataModule, DataModuleConfig
import torch
import numpy as np
import open3d as o3d
import torch
import pandas as pd
import torch_geometric
import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import FaceToEdge
import shutil
import torchvision.transforms as transforms
from loaders.factory import register
from dataclasses import dataclass
import trimesh


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


@dataclass
class ManifoldDataModuleConfig(DataModuleConfig):
    module: str = "datasets.manifold"
    num_samples: int = 100


def read_ply(path):
    mesh = trimesh.load_mesh(path)
    pos = torch.from_numpy(mesh.vertices).to(torch.float)
    face = torch.from_numpy(mesh.faces).to(torch.long).t()
    return Data(pos=pos, face=face)


class ManifoldDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([FaceToEdge(), CenterTransform()])
        super().__init__(
            config.root, config.batch_size, config.num_workers, config.pin_memory
        )

    def prepare_data(self):
        self.train_ds = ManifoldDataset(
            self.config, split="train", pre_transform=self.transform
        )
        self.test_ds = ManifoldDataset(
            self.config, split="test", pre_transform=self.transform
        )
        self.val_ds = ManifoldDataset(
            self.config, split="val", pre_transform=self.transform
        )
        self.entire_ds = ManifoldDataset(
            self.config, split="train", pre_transform=self.transform
        )

    def setup(self):
        pass


class ManifoldDataset(Dataset):
    """Represents a 2D segmentation dataset.

    Input params:
        configuration: Configuration dictionary.
    """

    def __init__(self, config, split, pre_transform):
        super().__init__(
            root=config.root,
            transform=pre_transform,
            pre_transform=pre_transform,
            pre_filter=None,
        )
        self.config = config
        self.split = split
        self.clean()
        self.files = []
        self.create_spheres()
        self.create_mobius()
        self.create_torus()
        self.file_frame = pd.DataFrame(self.files, columns=["filename", "y"])

    def clean(self):
        path = f"{self.config.root}/manifold/{self.split}"
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    def get(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        file_name = self.file_frame.iloc[index, 0]
        y = self.file_frame.iloc[index, 1]
        data = read_ply(file_name)
        data.x = data.pos
        data.pos = None
        data.y = torch.tensor([y])
        return data

    def len(self):
        # return the size of the dataset
        return len(self.file_frame)

    def create_spheres(self, noise=None):
        if not noise:
            noise = 0.1

        for i in range(self.config.num_samples):
            base_mesh = o3d.geometry.TriangleMesh.create_sphere()
            vertices = np.asarray(base_mesh.vertices)
            vertices += np.random.uniform(0, noise, size=vertices.shape)
            base_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            base_mesh.compute_vertex_normals()
            f_name = (
                f"{self.config.root}/manifold/{self.split}/sphere_{self.split}_{i}.ply"
            )
            o3d.io.write_triangle_mesh(f_name, base_mesh)
            self.files.append([f_name, int(0)])

    def create_mobius(self, noise=None):
        if not noise:
            noise = 0.1
        for i in range(self.config.num_samples):
            base_mesh = o3d.geometry.TriangleMesh.create_mobius()
            vertices = np.asarray(base_mesh.vertices)
            vertices += np.random.uniform(0, noise, size=vertices.shape)
            base_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            base_mesh.compute_vertex_normals()
            f_name = (
                f"{self.config.root}/manifold/{self.split}/mobius_{self.split}_{i}.ply"
            )
            o3d.io.write_triangle_mesh(f_name, base_mesh)
            self.files.append([f_name, int(1)])

    def create_torus(self, noise=None):
        if not noise:
            noise = 0.1

        for i in range(self.config.num_samples):
            base_mesh = o3d.geometry.TriangleMesh.create_torus()
            vertices = np.asarray(base_mesh.vertices)
            vertices += np.random.uniform(0, noise, size=vertices.shape)
            base_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            base_mesh.compute_vertex_normals()
            f_name = (
                f"{self.config.root}/manifold/{self.split}/torus_{self.split}_{i}.ply"
            )
            o3d.io.write_triangle_mesh(f_name, base_mesh)
            self.files.append([f_name, int(2)])


def initialize():
    register("dataset", ManifoldDataModule)

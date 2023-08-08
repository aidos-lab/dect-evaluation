from datasets.base_dataset import DataModule
import torch
import numpy as np
import open3d as o3d
import torch
import pandas as pd
import torch_geometric
import os
import torch
from torch_geometric.data import Dataset
from torch_geometric.transforms import FaceToEdge
import shutil
import torchvision.transforms as transforms


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


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
        path = f"{self.config.root}/{self.split}"
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    def get(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        file_name = self.file_frame.iloc[index, 0]
        y = self.file_frame.iloc[index, 1]
        data = torch_geometric.io.read_ply(file_name)
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
            f_name = f"{self.config.root}/{self.split}/sphere_{self.split}_{i}.ply"
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
            f_name = f"{self.config.root}/{self.split}/mobius_{self.split}_{i}.ply"
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
            f_name = f"{self.config.root}/{self.split}/torus_{self.split}_{i}.ply"
            o3d.io.write_triangle_mesh(f_name, base_mesh)
            self.files.append([f_name, int(2)])


# bunny = o3d.data.BunnyMesh()
""" gt_mesh = o3d.io.read_triangle_mesh(bunny.path)  """
""" #gt_mesh = o3d.geometry.TriangleMesh.create_torus() """


""" gt_mesh.compute_vertex_normals() """


""" print('create noisy mesh') """
""" noise = 0.1 """
""" vertices += np.random.uniform(0, noise, size=vertices.shape) """
""" mesh_in.vertices = o3d.utility.Vector3dVector(vertices) """
""" mesh_in.compute_vertex_normals() """
""""""
""""""
""" o3d.visualization.draw_geometries([mesh_in]) """
""""""


""" xyz = sample_from_sphere(n=100, d=2, r=1, noise=None, ambient=None, seed=None).detach().numpy() """


""" # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize """
""" pcd = o3d.geometry.PointCloud() """
""" pcd.points = o3d.utility.Vector3dVector(xyz) """
""""""
""" o3d.io.write_point_cloud("sync.ply", pcd) """

""""""
""" # Load saved point cloud and visualize it """
""" pcd_load = o3d.io.read_point_cloud("sync.ply") """
""" o3d.visualization.draw_geometries([pcd]) """

""" pcd = o3d.io.read_point_cloud("pointcloud.ply") """

""" pcd.estimate_normals() """
""""""
""" # estimate radius for rolling ball """
""" distances = pcd.compute_nearest_neighbor_distance() """
""" avg_dist = np.mean(distances) """
""" radius = 1.5 * avg_dist    """
""" print(radius) """

""" mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting( """
"""            pcd, """
"""            o3d.utility.DoubleVector([1, 20])) """

""" mesh,d = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson( """
"""            pcd, """
"""            depth=8, """
"""            width=0) """


""" mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20) """
""""""
""" knot_mesh = o3d.data.KnotMesh() """
""" mesh = o3d.io.read_triangle_mesh(knot_mesh.path) """
""" mesh.compute_vertex_normals() """

""" # create the triangular mesh with the vertices and faces from open3d """
""" tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), """
"""                           vertex_normals=np.asarray(mesh.vertex_normals)) """


""" o3d.visualization.draw_geometries([mesh])  """
""""""
""""""
""" trimesh.convex.is_convex(tri_mesh) """

import open3d as o3d
import numpy as np


path = "./data/modelnet_40_1000/raw/airplane/train/airplane_0001.off"


mesh = o3d.io.read_triangle_mesh(path)
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
vertices -= np.mean(vertices)

norm = np.sqrt(np.power(vertices, 2).sum(axis=1).max())
vertices /= norm
vertices -= np.repeat(np.array([[0.05, -0.4, 0.1]]), vertices.shape[0], axis=0)


# # vertices = np.divide(vertices.T, norm).T

# print(vertices)
# print(vertices.shape)
# print(vertices.mean())

mesh.vertices = o3d.utility.Vector3dVector(vertices)

o3d.io.write_triangle_mesh("test.ply", mesh)

from vedo import *
import vedo

sphere = vedo.shapes.Sphere(pos=(0, 0, 0), r=1, alpha=0.2)

g = load("test.ply")
show([g, sphere], axes=0)

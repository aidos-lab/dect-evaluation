import open3d as o3d
# bunny = o3d.data.BunnyMesh()
""" gt_mesh = o3d.io.read_triangle_mesh(bunny.path)  """
""" #gt_mesh = o3d.geometry.TriangleMesh.create_torus() """
mesh_in = o3d.geometry.TriangleMesh.create_sphere()
""" gt_mesh.compute_vertex_normals() """

import torch
import numpy as np

print('create noisy mesh')
vertices = np.asarray(mesh_in.vertices)
noise = 0.1
vertices += np.random.uniform(0, noise, size=vertices.shape)
mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
mesh_in.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_in])



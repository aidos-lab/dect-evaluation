import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import matplotlib.pyplot as plt
import open3d as o3d


def plot_batch(data):
    coords = data.x.cpu().numpy()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    sequence_containing_x_vals = coords[:, 0]
    sequence_containing_y_vals = coords[:, 1]
    sequence_containing_z_vals = coords[:, 2]

    ax.scatter(
        sequence_containing_x_vals,
        sequence_containing_y_vals,
        sequence_containing_z_vals,
    )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()


def compute_mean(data: Data) -> torch.Tensor:
    return data.x.mean(axis=0)


def compute_radius(data: Data) -> torch.Tensor:
    return data.x.pow(2).sum(axis=1).sqrt().max()


class SimplifyMesh:
    def __call__(self, data):
        mesh_in = o3d.geometry.TriangleMesh()
        mesh_in.triangles = o3d.utility.Vector3iVector(data.face.numpy().T)
        mesh_in.vertices = o3d.utility.Vector3dVector(data.pos.numpy())
        mesh_in.compute_vertex_normals()
        voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 64
        mesh_smp = mesh_in.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        print(torch.tensor(mesh_smp.triangles, dtype=torch.long).shape)
        return Data(
            x=torch.tensor(mesh_smp.vertices, dtype=torch.float32),
            face=torch.tensor(mesh_smp.triangles, dtype=torch.long).T,
            y=data.y,
        )


class CenterTransform(object):
    """
    This transform subtracts the mean per axis and scales the pointcloud to
    have unit radius.
    """

    def __call__(self, data):
        data.x -= compute_mean(data)
        data.x /= compute_radius(data)
        return data


# class Standardize(object):
#     """
#     NOT TESTED SHOULD BE CHECKED!!!
#     """

#     def __init__(self, samplepoints):
#         self.samplepoints = samplepoints

#     def __call__(self, data):
#         clipper = torch.mean(torch.abs(x))
#         z = torch.clip(x, -100 * clipper, 100 * clipper)
#         mean = torch.mean(z)
#         std = torch.std(z)
#         normalized = (z - mean) / std
#         data.x = normalized
#         return data


if __name__ == "__main__":
    """
    Important note, when testing.
    - The return type is a reference to the original transformed object,
        NOT a deep copy. Hence NEVER reuse data instances when testing.
    """
    # Test subtract of mean
    data = Data(x=torch.tensor([[20.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    data.x -= compute_mean(data)
    data.x /= compute_radius(data)
    print("subtract mean\n", data.x)
    plot_batch(data)

    # Test subtract of mean all
    data = Data(x=torch.tensor([[20.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    data.x -= data.x.mean()
    data.x /= compute_radius(data)
    print("subtract total mean\n", data.x)
    plot_batch(data)

    # Test normalize
    data = Data(x=torch.tensor([[2.0, 0, 0], [0, 1.0, 1.0], [0, 0, 1.0]]))
    data.x /= compute_radius(data)
    print("normalized\n", data.x)

    # Test center transform.
    data = Data(x=torch.tensor([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    transform = CenterTransform()
    out = transform(data)
    print("Center transform\n", out.x)

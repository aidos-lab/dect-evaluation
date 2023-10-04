import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import matplotlib.pyplot as plt
import open3d as o3d
import torchvision


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


class ThresholdTransform(object):
    def __call__(self, data):
        data.x = torch.hstack([data.pos, data.x])
        return data


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


class Normalize(object):
    def __call__(self, data):
        mean = data.x.mean()
        std = data.x.std()
        data.x = (data.x - mean) / std
        return data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class NCI109Transform(object):
    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float).unsqueeze(0).T
        atom_number = torch.argmax(data.x, dim=-1, keepdim=True)
        data.x = torch.hstack([deg, atom_number])
        # print(deg)
        # print(atom_number)
        # print()
        return data


class ModelNetTransform(object):
    def __call__(self, data):
        data.x = data.pos
        data.pos = None
        return data


class FloatTransform(object):
    def __call__(self, data):
        data.x = data.x.to(torch.float64)
        return data


class Rotate(object):
    def __call__(self, batch):
        theta = (torch.rand(1) - 0.5) * torch.pi / 5
        rot = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        batch.x = batch.x @ rot
        # scaling
        return batch


class Project(object):
    def __call__(self, batch):
        batch.x = batch.x[:, :2]
        # scaling
        return batch


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


class MnistTransform:
    def __init__(self):
        xcoords = torch.linspace(-0.5, 0.5, 28)
        ycoords = torch.linspace(-0.5, 0.5, 28)
        self.X, self.Y = torch.meshgrid(xcoords, ycoords)
        self.tr = torchvision.transforms.ToTensor()

    def __call__(self, data: tuple) -> Data:
        img, y = data
        img = self.tr(img)
        idx = torch.nonzero(img.squeeze(), as_tuple=True)
        gp = torch.vstack([self.X[idx], self.Y[idx]]).T
        dly = vedo.delaunay2d(gp, mode="xy", alpha=0.03).c("w").lc("o").lw(1)
        # print(torch.tensor(dly.edges()).T.shape)
        # print(torch.tensor(dly.faces()).T.shape)
        # print(torch.tensor(dly.points()).shape)

        return Data(
            x=torch.tensor(dly.points()),
            face=torch.tensor(dly.faces(), dtype=torch.long).T,
            y=torch.tensor(y, dtype=torch.long),
        )


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

from datasets.modelnet import ModelNetDataModuleConfig, ModelNetPointsDataModule
import h5py

config = ModelNetDataModuleConfig(
    module="datasets.modelnet",
    root="./data/modelnet_aligned_40_1000",
    samplepoints=10000,
    name="40",
)

dataset = ModelNetPointsDataModule(config)
print(dataset.entire_ds[0])


train_cloud = dataset.entire_ds.pos.view(9843, 10000, 3)
test_cloud = dataset.test_ds.pos.view(2468, 10000, 3)
train_labels = dataset.entire_ds.y
test_labels = dataset.test_ds.y


with h5py.File("ModelNet40_aligned_cloud.h5", "w") as f:
    f["tr_cloud"] = train_cloud.numpy()
    f["tr_labels"] = train_labels.numpy()
    f["test_cloud"] = test_cloud.numpy()
    f["test_labels"] = test_labels.numpy()

# import torch

# a = torch.range(1, 27 * 2)
# b = a.view(-1, 9, 3)
# c = b.view(-1, 3)
# print(c)

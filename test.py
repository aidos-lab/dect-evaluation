from datasets import load_datamodule
from omegaconf import OmegaConf
import open3d as o3d
import torch
import torch_geometric
import torch_geometric.transforms as tf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def main():
    config = OmegaConf.load("./config/torus_vs_spheres.yaml")
    dm = load_datamodule(
        name=config.data.name,
        config=config.data.config
    )

    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        print(batch.x)
        print(batch.y)
        break

if __name__ == "__main__":
    main()



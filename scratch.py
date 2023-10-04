import torch

# from torch_scatter import segment_coo

# src = torch.randn(10, 6, 64)
# index = torch.tensor([0, 0, 1, 1, 1, 2])
# index = index.view(1, -1)  # Broadcasting in the first and last dim.

# out = segment_coo(src, index, reduce="sum")

# print(out.size())


start = torch.tensor([1.0, 0.0])
l = torch.linspace(start, 2.0, 10)
print(l)

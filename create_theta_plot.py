from vedo import *
import torch
import vedo

thetas = torch.load("before.pt")
sphere = vedo.shapes.Sphere(pos=(0, 0, 0), r=1, alpha=0.2)

Points(thetas.cpu().detach().numpy()).show(axes=1).close()

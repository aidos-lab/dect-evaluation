# import os, sys
# import importlib
# from pydoc import locate
# from abc import ABC, abstractmethod
# import torch
# import torch.nn.functional as F
# from torch.utils.data import random_split

# """
# This package includes all the modules related to data loading and preprocessing.
# """

# """
# The below code fetches all files in the datasets folder and imports them. Each
# dataset can have its own file, or grouped in sets.
# """
# """ path = os.path.dirname(os.path.abspath(__file__)) """
# """ for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']: """
# """     mod = __import__('.'.join([__name__, py]), fromlist=[py]) """
# """     classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)] """
# """     for cls in classes: """
# """         setattr(sys.modules[__name__], cls.__name__, cls) """


# def load_model(name, config):
#     model = locate(f"models.{name}")
#     if not model:
#         print(f"Tried to load model {name}")
#         raise AttributeError()
#     return model(config)

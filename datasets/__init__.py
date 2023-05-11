import os, sys
import importlib
from pydoc import locate
import torch.utils.data as data
import torch_geometric

"""
This package includes all the modules related to data loading and preprocessing.
"""

"""
The below code fetches all files in the datasets folder and imports them. Each 
dataset can have its own file, or grouped in sets.
"""
path = os.path.dirname(os.path.abspath(__file__))
for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)


def get_dataset(dataset_name):
    dataset = locate(f'datasets.{dataset_name}')
    if not dataset:
        print(dataset_name)
        raise AttributeError()
    return dataset



class CustomDataLoader():
    """
    Wrapper class of Dataset class that performs multi-threaded data loading
    according to the configuration.

    NOTE: The torch_geometric dataloader class can handle both normal pytorch datasets and 
    graph datasets. 
    """
    def __init__(self, configuration):
        self.configuration = configuration
        dataset_class = get_dataset(configuration.name)
        self.dataset = dataset_class(configuration.dataset_params)
        self.dataloader = torch_geometric.loader.DataLoader(self.dataset, **vars(configuration.dataloader_params))

    def __len__(self):
        """Return the number of data in the dataset.
        """
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data.
        """
        for data in self.dataloader:
            yield data

    def info(self):
        self.testloader = torch_geometric.loader.DataLoader(self.dataset, **vars(self.configuration.dataloader_params))
        info = SimpleNamespace()
        for batch in self.testloader:
            break
        info.batch = batch
        return info





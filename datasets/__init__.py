import os, sys
from pydoc import locate

"""
This package includes all the modules related to data loading and preprocessing.
"""

"""
The below code fetches all files in the datasets folder and imports them. Each 
dataset can have its own file, or grouped in sets.
"""
""" path = os.path.dirname(os.path.abspath(__file__)) """
""" for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']: """
"""     mod = __import__('.'.join([__name__, py]), fromlist=[py]) """
"""     classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)] """
"""     for cls in classes: """
"""         setattr(sys.modules[__name__], cls.__name__, cls) """
""""""
""""""
def load_datamodule(name=None,config=None):
    dataset = locate(f'name')
    print("loading", dataset)
    if not dataset:
        print(name)
        raise AttributeError()
    return dataset(config)




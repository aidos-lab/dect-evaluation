from torch_geometric.datasets import TUDataset


#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯

"""
Add all the required transforms in this section, or use imports.
"""

#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯

"""
Define the dataset classes, provide dataset/dataloader parameters 
in the config file or overwrite them in the class definition.
"""

class TU_DATASET():
    def __init__(self,config):
        """
        This is the "flexible" base class for testing, for the experiments we 
        fix a set of parameters in the config file and run the experiment.
        """
        extra_config = {"name":"Letter-high"}
        self.dataset=TUDataset(**vars(config)|extra_config)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset.__getitem__(idx)




class PROTEINS_DATASET():
    def __init__(self,config):
        extra_config = {"name":"PROTEINS"}
        self.dataset=TUDataset(**vars(config)|extra_config)
        #config.pre_transform = transforms.Compose([ThresholdTransform()])
        
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset.__getitem__(idx)

class BRZ_DATASET():
    def __init__(self,config):
        extra_config = {"name":"BRZ"}
        self.dataset=TUDataset(**vars(config)|extra_config)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset.__getitem__(idx)

class LETTER_HIGH_DATASET():
    def __init__(self,config):
        extra_config = {"name":"Letter-high"}
        self.dataset=TUDataset(**vars(config)|extra_config)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset.__getitem__(idx)


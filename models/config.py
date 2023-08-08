from dataclasses import dataclass

@dataclass
class ECTModelConfig:
    module : str
    num_thetas  : int = 25
    bump_steps  : int = 25
    batch_size  : int = 128
    R           : float = 1.1
    scale       : int = 500
    num_features: int = 3
    num_classes : int = 10
    hidden      : int = 100

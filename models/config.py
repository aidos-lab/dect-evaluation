from dataclasses import dataclass


@dataclass(frozen=True)
class EctConfig:
    num_thetas: int = 64
    bump_steps: int = 64
    R: float = 1.1
    ect_type: str = "points"
    device: str = 'cpu'
    num_features: int = 3


@dataclass(frozen=True)
class ModelConfig:
    ectconfig: EctConfig = EctConfig()
    num_features: int = 3
    num_classes: int = 10
    hidden: int = 50

from dataclasses import dataclass


@dataclass(frozen=True)
class EctConfig:
    num_thetas: int = 32
    bump_steps: int = 32
    R: float = 1.1
    ect_type: str = "points"


@dataclass(frozen=True)
class ModelConfig:
    module: str
    ectconfig: EctConfig = EctConfig()
    num_features: int = 3
    num_classes: int = 10
    hidden: int = 50

from typing import Any, Protocol
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    meta: Any
    data: Any
    model: Any
    trainer: Any


@dataclass
class Meta:
    name: str
    project: str = "desct"
    tags: list[str] = field(default_factory=list)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


#  ╭──────────────────────────────────────────────────────────╮
#  │ Model Configurations                                     │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class ModelConfig:
    name: str
    config: Any


#  ╭──────────────────────────────────────────────────────────╮
#  │ Trainer configurations                                   │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class TrainerConfig:
    lr: float = 0.001
    num_epochs: int = 200

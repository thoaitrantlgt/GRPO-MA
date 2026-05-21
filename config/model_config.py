from dataclasses import dataclass

from trl import ModelConfig


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    name: str
    grid_search_params: dict


@dataclass
class GridSearchConfig:
    models: List[ModelConfig]

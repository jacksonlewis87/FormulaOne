from dataclasses import dataclass


@dataclass
class DataConfig:
    r_file: str
    cs_file: str
    ds_file: str
    min_season: int
    max_season: int

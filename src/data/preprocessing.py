import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from typing import Tuple

from data.data_config import DataConfig


def preprocess_dataset(df: pd.DataFrame, data_config: DataConfig) -> Tuple[pd.DataFrame, ...]:
    df = df[(df["season"] >= data_config.min_season) & (df["season"] <= data_config.max_season)]
    df = df.fillna(0.0)

    # randomly drop a percentage of 0 points scored samples
    np.random.seed(42)
    zero_rows = df[df["points"] == 0]
    rows_to_drop = zero_rows.sample(frac=0.85).index
    df = df.drop(index=rows_to_drop)

    point_columns = [
        "season_to_date_constructor_points_per_round",
        "season_to_date_driver_points_per_round",
        "previous_constructor_points_per_round",
        "previous_driver_points_per_round",
    ]

    x = df[["round", "grid"] + point_columns]
    y = df[["points"]]
    ids = df[["season", "round", "driver_id"]]

    # normalize
    pt = PowerTransformer(method="yeo-johnson")
    x.loc[:, point_columns] = pt.fit_transform(x[point_columns])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    y.loc[:, "points"] = y["points"].clip(upper=26)
    y.loc[:, "points"] = np.log((y["points"] / 26.0) + 0.1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = pd.DataFrame(scaler.fit_transform(y), columns=y.columns)
    y = y["points"]

    return x, y, ids

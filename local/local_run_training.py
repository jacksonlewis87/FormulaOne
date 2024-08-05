from constants import ROOT_DIR
from data.data_config import DataConfig
from model.entrypoint import run_training
from model.model_types import ModelType


def main():
    r_file = "results.csv"
    cs_file = "constructor_standings.csv"
    ds_file = "driver_standings.csv"

    run_training(
        data_config=DataConfig(
            r_file=f"{ROOT_DIR}/data/raw_data/{r_file}",
            cs_file=f"{ROOT_DIR}/data/raw_data/{cs_file}",
            ds_file=f"{ROOT_DIR}/data/raw_data/{ds_file}",
            min_season=2010,
            max_season=2020,
        ),
        model_type=ModelType.LGBM.value,
        model_params={
            "num_leaves": 4,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 60,
            "min_child_samples": 10,
            "verbosity": -1,
        },
    )


if __name__ == "__main__":
    main()

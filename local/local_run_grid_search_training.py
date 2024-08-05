from constants import ROOT_DIR
from data.data_config import DataConfig
from model.grid_search.entrypoint import grid_search_entrypoint
from model.grid_search.grid_search_config import GridSearchConfig, ModelConfig
from model.model_types import ModelType


def main():
    r_file = "results.csv"
    cs_file = "constructor_standings.csv"
    ds_file = "driver_standings.csv"

    grid_search_entrypoint(
        data_config=DataConfig(
            r_file=f"{ROOT_DIR}/data/raw_data/{r_file}",
            cs_file=f"{ROOT_DIR}/data/raw_data/{cs_file}",
            ds_file=f"{ROOT_DIR}/data/raw_data/{ds_file}",
            min_season=2010,
            max_season=2020,
        ),
        grid_search_config=GridSearchConfig(
            models=[
                ModelConfig(
                    name=ModelType.DECISION_TREE.value,
                    grid_search_params={
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    },
                ),
                ModelConfig(
                    name=ModelType.ELASTIC_NET.value,
                    grid_search_params={"alpha": [0.1, 1, 10, 100], "l1_ratio": [0.1, 0.5, 0.9]},
                ),
                ModelConfig(
                    name=ModelType.GRADIENT_BOOSTING.value,
                    grid_search_params={
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                    },
                ),
                ModelConfig(
                    name=ModelType.KNN.value,
                    grid_search_params={
                        "n_neighbors": [5, 9, 15, 21, 27],
                        "weights": ["uniform", "distance"],
                        "p": [1, 2],  # 1 is Manhattan distance, 2 is Euclidean distance
                    },
                ),
                ModelConfig(
                    name=ModelType.LASSO.value,
                    grid_search_params={"alpha": [0.1, 1, 10, 100, 1000]},
                ),
                ModelConfig(
                    name=ModelType.LGBM.value,
                    grid_search_params={
                        "boosting_type": ["gbdt", "dart", "goss"],
                        "num_leaves": [4, 5, 6],
                        "max_depth": [-1],  # -1 means no limit
                        "learning_rate": [0.05, 0.1, 0.2],
                        "n_estimators": [60, 70, 80],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                        "min_child_samples": [10, 20, 30],
                        "reg_alpha": [0, 0.1, 0.5, 1],
                        "reg_lambda": [0, 0.1, 0.5, 1],
                        "verbosity": [-1],
                    },
                ),
                ModelConfig(
                    name=ModelType.LINEAR.value,
                    grid_search_params={},
                ),
                ModelConfig(
                    name=ModelType.RANDOM_FOREST.value,
                    grid_search_params={
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    },
                ),
                ModelConfig(
                    name=ModelType.RIDGE.value,
                    grid_search_params={"alpha": [0.1, 1, 10, 100, 1000]},
                ),
                ModelConfig(
                    name=ModelType.SVR.value,
                    grid_search_params={
                        "kernel": ["linear", "poly", "rbf"],
                        "C": [0.1, 1, 10],
                        "gamma": ["scale", "auto"],
                    },
                ),
                ModelConfig(
                    name=ModelType.XGBOOST.value,
                    grid_search_params={
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 0.9, 1.0],
                    },
                ),
            ]
        ),
    )


if __name__ == "__main__":
    main()

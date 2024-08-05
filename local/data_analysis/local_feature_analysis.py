from constants import ROOT_DIR
from data.analysis.features import analyze_features
from data.data_config import DataConfig
from data.dataset import get_dataset
from data.preprocessing import preprocess_dataset


def main():
    r_file = "results.csv"
    cs_file = "constructor_standings.csv"
    ds_file = "driver_standings.csv"

    data_config = DataConfig(
        r_file=f"{ROOT_DIR}/data/raw_data/{r_file}",
        cs_file=f"{ROOT_DIR}/data/raw_data/{cs_file}",
        ds_file=f"{ROOT_DIR}/data/raw_data/{ds_file}",
        min_season=2001,
        max_season=2020,
    )

    dataset = get_dataset(
        r_file=data_config.r_file,
        cs_file=data_config.cs_file,
        ds_file=data_config.ds_file,
    )
    x, y, _ = preprocess_dataset(df=dataset, data_config=data_config)  # ignore ids

    analyze_features(x=x, y=y)


if __name__ == "__main__":
    main()

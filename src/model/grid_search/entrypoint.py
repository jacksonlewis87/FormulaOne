from data.data_config import DataConfig
from data.dataset import get_dataset
from data.preprocessing import preprocess_dataset
from model.grid_search.grid_search_driver import run_grid_search
from model.grid_search.grid_search_config import GridSearchConfig
from model.model_types import get_model_type


def grid_search_entrypoint(data_config: DataConfig, grid_search_config: GridSearchConfig):
    dataset = get_dataset(
        r_file=data_config.r_file,
        cs_file=data_config.cs_file,
        ds_file=data_config.ds_file,
    )

    x, y, _ = preprocess_dataset(df=dataset, data_config=data_config)  # ignore ids while training

    for model_config in grid_search_config.models:
        print(model_config.name)
        model = get_model_type(model_type=model_config.name)
        run_grid_search(model=model, params=model_config.grid_search_params, x=x, y=y)

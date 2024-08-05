from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from data.data_config import DataConfig
from data.dataset import get_dataset
from data.preprocessing import preprocess_dataset
from model.model_types import get_model_type
from model.shap_driver import run_shap


def run_training(data_config: DataConfig, model_type: str, model_params: dict):
    dataset = get_dataset(
        r_file=data_config.r_file,
        cs_file=data_config.cs_file,
        ds_file=data_config.ds_file,
    )

    x, y, _ = preprocess_dataset(df=dataset, data_config=data_config)  # ignore ids while training

    model = get_model_type(model_type=model_type)
    model.set_params(**model_params)

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    print(f"Train MSE: {round(mean_squared_error(y_train, y_pred), 3)}")
    print(f"Train R-Squared: {round(r2_score(y_train, y_pred), 3)}")

    y_pred = model.predict(X_test)
    print(f"Test MSE: {round(mean_squared_error(y_test, y_pred), 3)}")
    print(f"Test R-Squared: {round(r2_score(y_test, y_pred), 3)}")

    run_shap(model=model, x=x)

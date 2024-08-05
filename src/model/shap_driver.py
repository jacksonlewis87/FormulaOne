import shap
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any


def run_shap(model: Any, x: pd.DataFrame):
    explainer = shap.Explainer(model, x)

    # Compute SHAP values
    shap_values = explainer(x)

    # 1. Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x)
    plt.show()

    # 2. Dependence Plot
    for feature in [
        "round",
        "grid",
        "season_to_date_constructor_points_per_round",
        "season_to_date_driver_points_per_round",
        "previous_constructor_points_per_round",
        "previous_driver_points_per_round",
    ]:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, x)
        plt.show()

    # 3. Force Plot
    # Plot the force plot for the first instance in the dataset
    # Note: For this plot, you might need to use a Jupyter notebook or IPython environment
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], x.iloc[0])
    plt.show()

    # 4. Waterfall Plot
    # Plot the waterfall plot for the first instance in the dataset
    shap.waterfall_plot(shap_values[0])
    plt.show()

    # 5. Decision Plot
    # Plot the decision plot for the first instance in the dataset
    shap.decision_plot(explainer.expected_value, shap_values, x)
    plt.show()

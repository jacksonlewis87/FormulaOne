from enum import Enum
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class ModelType(Enum):
    DECISION_TREE = "decision_tree"
    ELASTIC_NET = "elastic_net"
    GRADIENT_BOOSTING = "gradient_boosting"
    KNN = "knn"
    LASSO = "lasso"
    LGBM = "lgbm"
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    RIDGE = "ridge"
    SVR = "svr"
    XGBOOST = "xgboost"


def get_model_type(model_type: str):
    if model_type == ModelType.DECISION_TREE.value:
        return DecisionTreeRegressor()
    if model_type == ModelType.ELASTIC_NET.value:
        return ElasticNet()
    if model_type == ModelType.GRADIENT_BOOSTING.value:
        return GradientBoostingRegressor()
    if model_type == ModelType.KNN.value:
        return KNeighborsRegressor()
    if model_type == ModelType.LASSO.value:
        return Lasso()
    if model_type == ModelType.LGBM.value:
        return LGBMRegressor()
    if model_type == ModelType.LINEAR.value:
        return LinearRegression()
    if model_type == ModelType.RANDOM_FOREST.value:
        return RandomForestRegressor()
    elif model_type == ModelType.RIDGE.value:
        return Ridge()
    elif model_type == ModelType.SVR.value:
        return SVR()
    elif model_type == ModelType.XGBOOST.value:
        return XGBRegressor()
    else:
        return None

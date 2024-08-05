from sklearn.model_selection import GridSearchCV


def run_grid_search(model, params: dict, x, y):

    # Set up GridSearchCV with 10-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="neg_mean_squared_error",
        cv=10,
        verbose=1,
        n_jobs=-1,
    )

    # Fit GridSearchCV
    grid_search.fit(x, y)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score:.4f}")

    results = grid_search.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"Mean Test Score: {-mean_score:.2f}, Parameters: {params}")

import pandas as pd
import numpy as np


from bayesopt_sklearn import BayesianOptimizer

from data import (
    df_to_numpy,
    clean_data,
    variable_names,
    parameter_bounds,
    parameter_granularity,
)

if __name__ == "__main__":
    xdata = pd.read_json("input_parameters_database.json")
    ydata = pd.read_json("goal_parameters_database.json")

    # combine the two
    data, num_compounds = clean_data(xdata, ydata)
    x, y = df_to_numpy(data)

    # run bayesian optimization
    bounds = [parameter_bounds[k] for k in variable_order]
    granularity = [parameter_granularity[k] for k in variable_order]
    optimizer = BayesianOptimizer(
        X_train=x, y_train=y, bounds=bounds, granularity=granularity
    )
    optimizer.optimize()

    # Get next point to evaluate
    next_point = optimizer.suggest_next_point()
    print(f"Sugesting experiment:")
    for i in range(len(next_point)):
        print(f"{variable_names[i]}={next_point[i]}")

    # Add experimental observation
    # optimizer.observe(next_point, value)

    # you can save and load the optimizer if you want,
    # but for so few points it's not worth it
    # optimizer.save("optimizer.pkl")
    # optimizer = BayesianOptimizer.load("optimizer.pkl")

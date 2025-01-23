import torch
import pandas as pd
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize

from data import (
    parameter_bounds,
    buckets,
    variable_order,
    clean_data,
    df_to_numpy,
    individiual_to_mixture_concentrations,
)


# TODO: remove this and replace with actual training data
def get_toy_training_data():
    xdata = pd.read_json("input_parameters_database.json")
    ydata = pd.read_json("goal_parameters_database.json")

    # combine the two
    data, num_compounds = clean_data(xdata, ydata)

    # data is missing pH_regulation, so add 0 everywhere
    if "pH_regulation" not in data.columns:
        data["pH_regulation"] = 0.0
    if "Na2MoO4" not in data.columns:
        data["Na2MoO4"] = 0.2

    # manually clamp the concentrations to the bounds
    data["NiSO4"] = data["NiSO4"].apply(lambda x: np.clip(x, 0.04, 0.4))
    data["Na2MoO4"] = data["Na2MoO4"].apply(lambda x: np.clip(x, 0.04, 0.4))
    # apply to data
    data["liquid1"], data["liquid2"] = zip(
        *data.apply(
            lambda row: individiual_to_mixture_concentrations(
                row["NiSO4"], row["Na2MoO4"]
            ),
            axis=1,
        )
    )

    x, y = df_to_numpy(data)

    return np.vstack(x).astype(float), y


# TODO: remove this and replace with actual experiment
def toy_objective_function(x):
    # fake experiment
    # nice convex function
    return -(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2)


# Define the bounds for the variables (6 variables)
bounds = torch.tensor([parameter_bounds[v] for v in variable_order], dtype=torch.double)

# Constraint: liquid1 + liquid2 <= 1
inequality_constraints = [
    (
        # indices of the variables we want to constrain
        torch.tensor(
            [variable_order.index("liquid1"), variable_order.index("liquid2")],
            dtype=torch.long,
        ),
        # coefficients of the linear combination (weighted sum)
        torch.tensor([1.0, 1.0], dtype=torch.double),
        # smaller or equal to
        1.0,
    )
]


def get_my_gp(_x, _y):
    # Define the GP model
    return SingleTaskGP(
        train_X=_x,
        train_Y=_y,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=6, bounds=bounds.T),
    )


# Get training data
# torch.tensor of shape (num_samples, num_variables)
# TODO: remove this and replace with actual training data
train_X, train_Y = get_toy_training_data()
train_X = torch.from_numpy(train_X).double()
train_Y = torch.from_numpy(train_Y).double()

# check if training data is within bounds
for i in range(train_X.shape[1]):
    within_bounds = torch.all(train_X[:, i] >= float(bounds[i, 0])) and torch.all(
        train_X[:, i] <= float(bounds[i, 1])
    )
    if not within_bounds:
        print(
            f"Warning: Variable `{variable_order[i]}` of training data is not within bounds. "
            "Won't couse a crash, but will cause a warning inside botorch to use min-max scaling, "
            "and BayesOpt might not perform well. "
            f"\nvariable bounds: {bounds[i, 0]} <= {variable_order[i]} <= {bounds[i, 1]}"
            # f"\nvariable values: {train_X[:, i]}"
        )

# Train the GP model on the initial training data
gp = get_my_gp(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
mll = fit_gpytorch_mll(mll)

# Start the optimization loop with real experiments
max_iterations = 10  # TODO: set this to whatever you want
experiments_per_iteration = 3  # TODO: set this to whatever you want
for iteration in range(max_iterations):
    # Define the acquisition function
    qEI = qLogExpectedImprovement(model=gp, best_f=train_Y.max())

    # Optimize the acquisition function
    new_Xs, acq_values = optimize_acqf(
        acq_function=qEI,
        bounds=bounds.T,
        q=experiments_per_iteration,  # number of candidates
        inequality_constraints=inequality_constraints,
        num_restarts=10,
        raw_samples=50,
    )

    for new_X in new_Xs:
        # Evaluate the new point
        new_Y = toy_objective_function(new_X).unsqueeze(-1)

        # round to nearest bucket
        for i, v in enumerate(new_X):
            possible_values = buckets[variable_order[i]]
            # only round if the bucket is not -1 (continuous variable)
            if possible_values is not None:
                new_X[i] = possible_values[torch.argmin(torch.abs(possible_values - v))]

        print(f"New suggested experiment: {new_X}")
        # print(f'Expected value: {new_Y.item()}')

        # TODO:
        # save experiment to file
        # run experiment
        # get experimental Y value
        # get actual experimental X values

        # TODO: remove this and replace with actual experiment
        new_Y = toy_objective_function(new_X).unsqueeze(-1)

        # usually experimental parameters deviate from the intended values
        actual_X = new_X  # TODO: get actual experimental X values

        # Update the training data
        # (num_samples, num_variables)
        train_X = torch.cat([train_X, actual_X.unsqueeze(0)], dim=0)
        # (num_samples, 1)
        train_Y = torch.cat([train_Y, new_Y.unsqueeze(0)], dim=0)

    # Refit the GP model with the new experimental data
    gp = get_my_gp(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll = fit_gpytorch_mll(mll)

# Print final results
print("Optimized Input:", train_X[train_Y.argmax()])
print("Optimized Objective Value:", train_Y.max().item())

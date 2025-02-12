import torch
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy, qUpperConfidenceBound, qKnowledgeGradient, UpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples

from data import (
    parameter_bounds,
    buckets,
    variable_order,
    clean_data,
    df_to_numpy,
    individiual_to_mixture_concentrations,
)

#########################################################################
# Helper functions
#########################################################################

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

# plot the toy objective function with plotly
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = toy_objective_function(np.vstack([X, Y]))
fig = go.Figure(data=go.Contour(z=Z, x=x, y=y))
# save the figure
fig.write_image("toy_objective_function.png")


#########################################################################
# Main
#########################################################################

def run_bayesopt(acquisition_function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    max_iterations = 12  # TODO: set this to whatever you want
    experiments_per_iteration = 1  # TODO: set this to whatever you want
    enforce_bounds = True # TODO: set to False during experiments, only for toy data
    
    #########################################################################
    # Define variables, bounds, constraints
    #########################################################################

    # Define the bounds for the variables 
    bounds = torch.tensor([parameter_bounds[v] for v in variable_order], dtype=torch.double).to(device)

    # Constraint: liquid1 + liquid2 <= 1
    inequality_constraints = [
        (
        # indices of the variables we want to constrain
            torch.tensor(
                [variable_order.index("liquid1"), variable_order.index("liquid2")],
                dtype=torch.long,
            ).to(device),
            # coefficients of the linear combination (weighted sum)
            -1 * torch.tensor([1.0, 1.0], dtype=torch.double).to(device),
            # smaller or equal to
            -1.0,
        ),
        # add another constraint 
        # liquid1 + liquid2 >= 0.5
        (
            torch.tensor(
                [variable_order.index("liquid1"), variable_order.index("liquid2")],
                dtype=torch.long,
            ).to(device),
            torch.tensor([1.0, 1.0], dtype=torch.double).to(device),
            0.5,
        ),
    ]
    
    #########################################################################
    # Get training data
    #########################################################################
    # torch.tensor of shape (num_samples, num_variables)
    # TODO: remove this and replace with actual training data
    train_X, train_Y = get_toy_training_data()
    train_X = torch.from_numpy(train_X).double().to(device)
    train_Y = torch.from_numpy(train_Y).double().to(device)

    # BoTorch maximizes the objective function, so we need to negate the objective function
    train_Y = -train_Y
    
    # check if training data is within bounds
    for i in range(train_X.shape[1]):
        if enforce_bounds:
            # clamp the variable to the bounds
            train_X[:, i] = torch.clamp(train_X[:, i], bounds[i, 0], bounds[i, 1])
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

    #########################################################################
    # Train the GP model on the initial training data
    #########################################################################

    def get_my_gp(_x, _y):
        # Define the GP model
        return SingleTaskGP(
            train_X=_x,
            train_Y=_y,
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=6, bounds=bounds.T),
        ).to(device)
        
    gp = get_my_gp(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll = fit_gpytorch_mll(mll)
    
    #########################################################################
    # Run the optimization / experiment loop
    #########################################################################
    # for qMaxValueEntropy
    if bounds.shape[1] == 2:
        _bounds = bounds.T
    else:
        _bounds = bounds
    assert _bounds.shape[0] == 2, f"bounds.shape: {_bounds.shape}"
    assert _bounds.shape[1] == 6, f"bounds.shape: {_bounds.shape}"
    candidate_set = draw_sobol_samples(bounds=_bounds, n=10_000, q=1).squeeze(1)

    experimental_values = []

    # Start the optimization loop with real experiments
    for iteration in range(max_iterations):
        # Define the acquisition function
        if acquisition_function == "qMaxValueEntropy":
            acq_fnct = qMaxValueEntropy(model=gp, candidate_set=candidate_set)
        elif acquisition_function == "qUpperConfidenceBound":
            # beta: controls the trade-off between exploration and exploitation
            # 0: only exploitation, >3: strong exploration
            acq_fnct = qUpperConfidenceBound(model=gp, beta=4)
        elif acquisition_function == "qKnowledgeGradient":
            # num_fantasies: more is better but slower (default: 64)
            acq_fnct = qKnowledgeGradient(model=gp, num_fantasies=16)
        elif acquisition_function == "KnowledgeGradient":
            # num_fantasies: more is better but slower (default: 64)
            acq_fnct = KnowledgeGradient(model=gp, num_fantasies=16)
        elif acquisition_function == "qLogExpectedImprovement":
            acq_fnct = qLogExpectedImprovement(model=gp, best_f=train_Y.max())
        else:
            raise ValueError(f"Invalid acquisition function: {acquisition_function}")

        # Optimize the acquisition function
        new_Xs, acq_values = optimize_acqf(
            acq_function=acq_fnct,
            bounds=bounds.T,
            q=experiments_per_iteration,  # number of candidates
            inequality_constraints=inequality_constraints,
            num_restarts=10,
            raw_samples=50,
        )

        for new_X in new_Xs:
            # Evaluate the new point
            new_Y = toy_objective_function(new_X).unsqueeze(-1).to(device)

            # round to nearest bucket
            for i, v in enumerate(new_X):
                possible_values = buckets[variable_order[i]]
                # only round if the bucket is not -1 (continuous variable)
                if possible_values is not None:
                    if isinstance(possible_values, list):
                        possible_values = torch.tensor(possible_values, dtype=torch.double)
                    possible_values = possible_values.to(device)
                    new_X[i] = possible_values[torch.argmin(torch.abs(possible_values - v))]

            posterior = gp.posterior(new_X.unsqueeze(0))
            expected_y = posterior.mean.item()
            uncertainty = posterior.variance.item()
            print('-'*100)
            print(f"New suggested experiment: {new_X}")
            print(f'Expected value: {expected_y}')
            print(f'Uncertainty: {uncertainty}')

            # TODO:
            # save experiment to file
            # run experiment
            # get experimental Y value
            # get actual experimental X values

            # TODO: remove this and replace with actual experiment
            new_Y = toy_objective_function(new_X).unsqueeze(-1)
            experimental_values.append(new_Y.item())
            # BoTorch maximizes the objective function, so we need to negate the objective function
            new_Y = -new_Y
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
    best_idx = train_Y.argmax()
    print('='*100)
    print("Optimized Input:", train_X[best_idx])
    print("Optimized Objective Value:", train_Y[best_idx].item())
    print("Experimental values:", experimental_values)
    print("Objective got smaller by:", train_Y[best_idx].item() - experimental_values[0])
    print("Objective got smaller (bool):", experimental_values[0] > train_Y[best_idx].item())


if __name__ == "__main__":
    # from exploration to exploitation:
    # qMaxValueEntropy, qKnowledgeGradient, qUpperConfidenceBound, qLogExpectedImprovement
    acquisition_function = "qUpperConfidenceBound"
    run_bayesopt(acquisition_function)
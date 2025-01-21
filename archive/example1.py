# in this block i have created at ML model based on the initial data

import pandas as pd
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.pyplot as plt
import os
import time

# Define parameter space ranges


def denormaliser(data, normaliser_params, axis=None):
    """Denormalize data"""
    return data * normaliser_params["x_std"] + normaliser_params["x_mean"]


def normalize_inputs(inputs, normaliser_params=None, dtype=torch.float32):
    """Normalize data to have mean=0 and std=1. """
    inputs = torch.tensor(inputs, dtype=dtype)
    # If normaliser_params is None, compute the mean and std.
    if normaliser_params is None:
        x_mean, x_std = inputs.mean(dim=0), inputs.std(dim=0)
        normaliser_params = {"x_mean": x_mean, "x_std": x_std}
    else:
        x_mean, x_std = normaliser_params["x_mean"], normaliser_params["x_std"]
    normalized_x = (inputs - x_mean) / x_std
    return normalized_x, normaliser_params




def create_model(
    acquisition_fct: str,  # 'GIBBON' or 'UCB'
    x: torch.Tensor,
    num_candidates: int = 10,
    beta: float = 0.5,
    save_model_path="",
    iteration=0,
    y=[],
):
    """
    This function creates a model based on the input parameters and returns the best candidate

    Args:
        acquisition_fct (str): The acquisition_fct to use for the model. GIBBON=Max Value Entropy, UCB=Upper Confidence Bound. Defaults to "GIBBON".
        num_candidates (int, optional): The number of candidates to use. Defaults to 10.
        beta (float, optional): The beta value to use. Defaults to 0.5.
        save_model_path (str, optional): The path to save the model. Defaults to "".
        iteration (int, optional): The iteration number. Defaults to 0.
        y (list, optional): The y data. Defaults to [].

    Raises:
        ValueError: If the acquisition_fct is not known.

    Returns:
        _type_: The best candidate.
    """

    # these are included for setting proper bounds
    # Our variables are
    # [current density (mA/cm2), deposition time [s],
    # pH regulation liquid 0-1.5 ml,
    # temperature
    # concentration of NiSO4 0.8, concentration of Mo 0.8]
    

    x = torch.tensor(x, dtype=torch.float32)
    normalized_x, normaliser_params_x = normalize_inputs(x)
    normalized_y, normaliser_params_y = normalize_inputs(y)
    
    # TODO: ?
    # We here need a way to normalize the stability slope for the y parameter
    y_activity_normalized = 1 - (abs(y[:, 1]) - 200) / (500 - 200)
    y_stability_normalized = 1 - (abs(y[:, 0])) / (0.4)

    w_stability = 0.5
    w_activity = 0.5
    y_sum_normalized = (
        w_activity * y_activity_normalized + w_stability * y_stability_normalized
    )
    y_sum_normalized = torch.tensor(y_sum_normalized, dtype=torch.float32)

    # Compute bounds
    # Insert new bounds

    bounds = torch.stack([normalized_x.min(dim=0)[0], normalized_x.max(dim=0)[0]])

    num_iterations = len(
        [
            file
            for file in os.listdir(os.path.abspath(save_model_path))
            if file.endswith(".pth")
        ]
    )

    # Train a dummy Gaussian Process model

    # Reshape normalized_y to be 2D
    y = y_sum_normalized.unsqueeze(-1)

    # here we remove the final two points such that we set the correct boundary
    normalized_x = normalized_x[:-2]
    # y=y[:-2]
    print(np.shape(normalized_x))
    print(np.shape(y))

    gp_model = SingleTaskGP(normalized_x, y)

    torch.save(
        {
            "model_state_dict": gp_model.state_dict(),  # Model parameters
            "normalized_x": normalized_x,  # Input data
            "normalized_y": y,  # Target data
            "normaliser_params": normaliser_params,  # Normalization parameters
        },
        os.path.join(
            save_model_path, f"gp_model_checkpoint_iteration_{num_iterations + 1}.pth"
        ),
    )

    ###############################################################################################

    # We need to incorporate the following constraints:
    # sum_i C_des_i / C_stock_i <= 1
    # C_des_i <= C_stock_i for all i
    # where C_des_i is the desired concentration of the i-th compound
    # and C_stock_i is the stock concentration of the i-th compound
    

    # Candidate selection logic
    if acquisition_fct == "GIBBON":
        # Create a candidate set
        candidate_set = torch.rand(1000, normalized_x.size(1))  # Large random set
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set

        # Define the acquisition function
        qGIBBON = qLowerBoundMaxValueEntropy(gp_model, candidate_set)

        # Optimize acquisition function
        candidate_norm, _ = optimize_acqf(
            qGIBBON,
            bounds=bounds,
            q=num_candidates,
            num_restarts=10,
            raw_samples=512,
            sequential=True,
        )

    elif acquisition_fct == "UCB":
        # Define the acquisition function
        UCB = UpperConfidenceBound(gp_model, beta=beta)

        # Optimize acquisition function
        candidate_norm, _ = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
    else:
        raise ValueError(f"Unknown acquisition_fct: {acquisition_fct}")

    # Denormalize candidates
    candidate, _ = denormaliser(
        candidate_norm, normaliser_params=normaliser_params
    )

    # Convert to DataFrame for better usability
    candidate_df = pd.DataFrame(
        candidate.numpy(), columns=[f"param{i+1}" for i in range(candidate.shape[1])]
    )

    return candidate_df



if __name__ == "__main__":

    """
    This is the main function that runs the Bayesian Optimization
    Input variables:
    discrete:
    - concentrations of n (1 to 3 for now) compounds. Granularity 1/200 * 0.8
    - temperature. Granularity 1 degree C
    - ph regulation. Granularity 1/20 * (0.3-0) or 21 including 0.3
    continuos:
    - deposition time
    - current density

    - what is the objective function? -> energy over the lifetime, integral (area under the curve)
    """

    # simple toy data with a single concentration
    # get parameters from json
    # optimize for lower slope (more negative slope)
    # build surrogate model and show that BayesOpt converges
    
    xdata = pd.read_json("input_parameters_database.json")
    ydata = pd.read_json("goal_parameters_database.json")
    
    # combine the two
    data, num_compounds = clean_data(xdata, ydata)
    x, y = df_to_torch(data, variable_names)


"""
Todo
- constraints in BayesOpt
"""

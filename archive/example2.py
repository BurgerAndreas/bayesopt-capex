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
from typing import Dict, Tuple


class ConstraintCreater:
    def __init__(self, normaliser_params: Dict[str, torch.tensor]):
        self.x_mean = normaliser_params["x_mean"]
        self.x_std = normaliser_params["x_std"]

    def to_gpr_space(self, b: float) -> Tuple[torch.tensor, float]:
        coefs_norm = self.x_std / (self.x_std.sum())
        b_norm = (b - self.x_mean.sum()) / (self.x_std.sum())
        return coefs_norm, b_norm

    def get_constraint_tuple(self, b: float, constraint_indices: list) -> tuple:
        constraint_indices = torch.tensor(constraint_indices)
        coefs_norm, b_norm = self.to_gpr_space(b)
        return (constraint_indices, coefs_norm, b_norm)


# Define parameter space ranges
def denormaliser(data, normaliser_params, axis):
    return data * normaliser_params["x_std"] + normaliser_params["x_mean"], None


def Create_Model(
    method: str,  # 'GIBBON' or 'UCB'
    x: torch.Tensor,
    num_candidates: int = 10,
    beta: float = 0.5,
    save_model_path="",
    iteration=0,
    y=[],
):
    # these are included for setting proper bounds
    # Our variables are [current density (mA/cm2), deposition time [s],temperature  concentration of NiSO4 0.8, concentration of Mo 0.8, pH regulation liquid 0-1.5 ml,]
    x_paramater_low = [1, 60, 30, 0, 0, 0]
    x_parameter_high = [200, 600, 80, 0.8, 0.8, 0.3]
    # x_data.append(x_paramater_low)
    # x_data.append(x_parameter_high)
    # print("initial",x)
    x = np.append(x, [x_paramater_low], axis=0)
    x = np.append(x, [x_parameter_high], axis=0)
    # print("new",x)
    x = torch.tensor(x, dtype=torch.float32)
    normalized_x, normalized_y, normaliser_params = normalize_inputs(
        y_parameter_sets=y, parameter_sets=x
    )

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
    # before  torch.rand(normalized_x.size(0), 1)
    # y = normalized_y  # Dummy output for GP
    # Reshape normalized_y to be 2D
    y = y_sum_normalized.unsqueeze(-1)
    # here we remove the final two points such that we set the correct boundary
    # print("before",np.shape(normalized_x))
    # print(normalized_x)
    normalized_x = normalized_x[:-2]
    # y=y[:-2]
    print(np.shape(normalized_x))
    print(np.shape(y))
    # print(normalized_x)
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
    # We need to incorporate the following constraint:
    # 1 - sum C_des / C_stock >= 0
    # constraints = [(to.tensor([2,4,6]),to.tensor([1,1,1]),1)]
    constrain = True
    if constrain:
        constrainer = ConstraintCreater(normaliser_params)
        # Specify constraint parameters
        constraint_indices = torch.tensor([0, 1, 2])
        constraint_coefs = torch.tensor(
            [1.0, 1.0, 1.0]
        )  #  OBS: not yet implemented in ConstraintCreater
        b = 400  # bound in physical space
        constraint_tuple = constrainer.get_constraint_tuple(b, constraint_indices)
    # Candidate selection logic
    if method == "GIBBON":
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
            equality_constraints=[constraint_tuple] if constrain else None,
        )
    elif method == "UCB":
        # Define the acquisition function
        UCB = UpperConfidenceBound(gp_model, beta=beta)
        # Optimize acquisition function
        candidate_norm, _ = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            equality_constraints=[constraint_tuple] if constrain else None,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    # Denormalize candidates
    candidate, _ = denormaliser(
        candidate_norm, normaliser_params=normaliser_params, axis=0
    )
    # Convert to DataFrame for better usability
    candidate_df = pd.DataFrame(
        candidate.numpy(), columns=[f"param{i+1}" for i in range(candidate.shape[1])]
    )
    return candidate_df


def normalize_inputs(y_parameter_sets, parameter_sets):
    # live data analysis
    y_data_orig = torch.tensor(y_parameter_sets, dtype=torch.float32)  # Your dataset
    y_mean, y_std = y_data_orig.mean(dim=0), y_data_orig.std(dim=0)
    normalized_y = (y_data_orig - y_mean) / y_std
    x_data_orig = torch.tensor(parameter_sets, dtype=torch.float32)  # Your dataset
    x_mean, x_std = x_data_orig.mean(dim=0), x_data_orig.std(dim=0)
    normalized_x = (x_data_orig - x_mean) / x_std
    normaliser_params = {"x_mean": x_mean, "x_std": x_std}
    return normalized_x, normalized_y, normaliser_params

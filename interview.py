import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy, qUpperConfidenceBound, qKnowledgeGradient, UpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples


# list of solvents:
solvents = ["water", "ethanol", "acetone", "toluene", "chloroform"]

# define five input variables with bounds
bounds = {
    'solvent': (0., 4.), # integer
    "temperature": (20., 100.), 
    # 'pH': (3., 8.),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# initial data is one random point
def experiment_selector(prev_xdatas, prev_ydatas):
    d = len(bounds)
    # Convert list of tensors to single tensor
    _xdatas = torch.stack(prev_xdatas).squeeze(1).to(dtype=dtype)
    _ydatas = torch.tensor(prev_ydatas, dtype=dtype).reshape(-1, 1)
    _bounds = torch.tensor([[b[0], b[1]] for b in bounds.values()], dtype=dtype) # [num_inputs, 2]
    # bounds have to be [2, d]
    if _bounds.shape[-1] == 2:
        _bounds = _bounds.T
    gp = SingleTaskGP(
        train_X=_xdatas, # [num_samples, d]
        train_Y=_ydatas, # [num_samples, 1]
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=d, bounds=_bounds),
    )#.to(device)
        
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll = fit_gpytorch_mll(mll)
    
    acq_fnct = qUpperConfidenceBound(model=gp, beta=2)
    
    # Optimize the acquisition function
    new_Xs, acq_values = optimize_acqf(
        acq_function=acq_fnct,
        bounds=_bounds,
        q=1,  # number of candidates
        num_restarts=10,
        raw_samples=50,
    )
    
    # round the solvent to the nearest integer
    new_Xs[:, 0] = torch.round(new_Xs[:, 0])
    
    # make prediction
    posterior = gp.posterior(new_Xs.unsqueeze(0))
    expected_y = posterior.mean.item()
    uncertainty = posterior.variance.item()
    return new_Xs, expected_y, uncertainty

def reaction_simulator(x_data):
    return torch.tensor([np.random.uniform(0, 1)], dtype=dtype)

if __name__ == "__main__":
    xdatas = []
    ydatas = []
    expected_ys = []
    uncertainties = []
    
    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # initial random data
    for _ in range(10):
        xdatas.append(
            torch.tensor(
                [np.random.randint(bounds[_b][0], bounds[_b][1]) for _b in bounds.keys()],
                dtype=dtype
            ).unsqueeze(0) # [1, d]
        )
        ydatas.append(reaction_simulator(xdatas[-1]).item())

    # optimization loop
    for iteration in range(1000):
        x_data, expected_y, uncertainty = experiment_selector(xdatas, ydatas)
        y_data = reaction_simulator(x_data).item()
        
        xdatas.append(x_data)
        ydatas.append(y_data)
        expected_ys.append(expected_y)
        uncertainties.append(uncertainty)
        
        print(f"\nIteration {iteration}:")
        print(f" Solvent: {solvents[int(x_data[0, 0])]}, Temperature: {int(x_data[0, 1])}")
        print(f' Expected yield: {expected_y:.2f}')
        print(f' Uncertainty: {uncertainty:.2f}')
        print(f" Reaction yield: {y_data:.2f}")
        
        # stop if y_data is > 95
        if y_data > 0.99:
            print("-"*100)
            print(f"Found solution at iteration {iteration}")
            break
        
        
    # plot heatmap with x_values solvents over temperature (bucket) , y_values as colors
    # Create temperature buckets (10 degree intervals)
    # temp_buckets = np.linspace(bounds['temperature'][0], bounds['temperature'][1], 9)

    # # Create 2D array for heatmap data
    # heatmap_data = np.zeros((len(solvents), len(temp_buckets)-1))
    # counts = np.zeros((len(solvents), len(temp_buckets)-1))

    # # Fill heatmap data by averaging y values for each solvent-temperature combination
    # for x, y in zip(xdatas, ydatas):
    #     solvent_idx = int(x[0][0])
    #     temp = x[1][0]
    #     temp_idx = np.digitize(temp, temp_buckets) - 1
    #     if temp_idx < len(temp_buckets)-1:  # Ensure we don't go out of bounds
    #         heatmap_data[solvent_idx][temp_idx] += y
    #         counts[solvent_idx][temp_idx] += 1

    # # Average the values where we have multiple data points
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     heatmap_data = np.divide(heatmap_data, counts)
    #     heatmap_data = np.nan_to_num(heatmap_data, 0)  # Replace NaN with 0

    # # Create labels for temperature buckets
    # temp_labels = [f"{int(temp_buckets[i])}-{int(temp_buckets[i+1])}°C" 
    #                for i in range(len(temp_buckets)-1)]

    # # Plot heatmap
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(heatmap_data, 
    #             xticklabels=temp_labels,
    #             yticklabels=solvents,
    #             cmap='viridis',
    #             annot=True,
    #             fmt='.2f',
    #             cbar_kws={'label': 'Reaction Yield'})

    # plt.xlabel('Temperature Range')
    # plt.ylabel('Solvent')
    # plt.title('Reaction Yield Heatmap')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('heatmap.png')


    # pick one solvent, plot the reaction yield over temperature















import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
import botorch
from botorch.utils.sampling import draw_sobol_samples

import sklearn
import umap



from plotting_helpers import *

current_dir = os.path.dirname(os.path.abspath(__file__))
plotfolder = os.path.join(current_dir, "plots")

# load data
data = pd.read_csv("analysis/data.csv", delimiter=";")

# rename the columns
data = data.rename(columns=variable_names)

ycol = "stability_slope"
xcols = [_c for _c in data.columns if _c != ycol]

# reorder the columns based on variable_order
data = data[variable_order + [ycol, "experiment"]]


############################################################################
# # What does the model believe are the best parameters?
############################################################################
# Plots to illustrate region of interest / relationships between variables
# Drawn from the posterior distribution (final belief of the model)


# final state of the model
xdatatensor = get_torch_from_df(data, variable_order)
ydatatensor = torch.tensor(data[ycol].values, dtype=torch.double).unsqueeze(-1)
best_gp = get_my_gp(xdatatensor, ydatatensor)


# pick parameters at best y value, lowest is best
best_data = data.sort_values(by="stability_slope", ascending=True)
best_xdata_tensor = get_torch_from_df(best_data, variable_order)
best_ydatatensor = torch.tensor(best_data[ycol].values, dtype=torch.double).unsqueeze(-1)


############################################################################
# Plot predicted mean (value) and uncertainty over variable
############################################################################
# plot predicted mean and uncertainty interval of one variable 
# while fixing the other variables at the best values
print('-'*80)
for var in variable_order:
    var_idx = variable_order.index(var)
    temps = torch.linspace(bounds[var_idx, 0], bounds[var_idx, 1], 100)
    best_inputs = best_data.iloc[0]
    means = []
    uncertainties = []
    for _t in temps:
        _input = best_xdata_tensor[0].clone()
        _input[var_idx] = _t
        posterior = best_gp.posterior(_input.unsqueeze(0))
        means.append(posterior.mean.item())
        uncertainties.append(posterior.variance.sqrt().item())
    fig = px.line(
        x=temps,
        y=means,
        error_y=uncertainties,
    )
    fig.update_layout(
        title=f"Predicted {human_names[ycol]} vs {human_names[var]}",
        xaxis_title=human_names[var],
        yaxis_title=f"Predicted {human_names[ycol]}",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),  # Remove whitespace around plot
    )
    fname = f"{plotfolder}/stability_slope_vs_{var}.png"
    fig.write_image(fname)
    print(f"Saved {fname}")
    # fig.show()



# # plot predicted mean and uncertainty interval of temp while fixing the other variables at the best values
# # same plot as above, but with uncertainty as shaded region
# print('-'*80)
# for var in variable_order:
#     var_idx = variable_order.index(var)
#     temps = torch.linspace(bounds[var_idx, 0], bounds[var_idx, 1], 100)
#     best_inputs = best_data.iloc[0]
#     means = []
#     uncertainties = []
#     for _t in temps:
#         _input = best_xdata_tensor[0].clone()
#         _input[var_idx] = _t
#         posterior = best_gp.posterior(_input.unsqueeze(0))
#         means.append(posterior.mean.item())
#         uncertainties.append(posterior.variance.sqrt().item())

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=temps,
#         y=means,
#         mode='lines',
#         name='Mean prediction'
#     ))
#     fig.add_trace(go.Scatter(
#         x=temps,
#         y=[m + c for m, c in zip(means, uncertainties)],
#         mode='lines',
#         line=dict(width=0),
#         showlegend=False
#     ))
#     fig.add_trace(go.Scatter(
#         x=temps,
#         y=[m - c for m, c in zip(means, uncertainties)],
#         mode='lines',
#         line=dict(width=0),
#         fillcolor='rgba(68, 134, 255, 0.3)',
#         fill='tonexty',
#         name='±1 std'
#     ))
#     fig.update_layout(
#         title=f"Predicted {human_names[ycol]} vs {human_names[var]}",
#         xaxis_title=human_names[var],
#         yaxis_title=f"Predicted {human_names[ycol]}",
#         showlegend=True,
#         margin=dict(l=0, r=0, t=30, b=0),  # Remove whitespace around plot
#     )
#     fig.show()

############################################################################
# Heatmaps
############################################################################


# # plot a heatmap of two variables, while fixing the other variables at the best values
# # color by the predicted stability slope
# print('-'*80)
# params_to_plot = [
#     ["temperature", "liquid2"],
#     ["temperature", "liquid1"],
#     ["current_density", "liquid2"],
# ]
# for params in params_to_plot:
#     npoints = 50
#     yvals = torch.linspace(bounds[variable_order.index(params[0]), 0], bounds[variable_order.index(params[0]), 1], npoints)
#     xvals = torch.linspace(bounds[variable_order.index(params[1]), 0], bounds[variable_order.index(params[1]), 1], npoints)
#     best_inputs = best_data.iloc[0]
#     means = np.zeros((len(yvals), len(xvals)))
#     uncertainties = np.zeros((len(yvals), len(xvals)))
#     for i, _y in enumerate(yvals):
#         for j, _x in enumerate(xvals):
#             _input = best_xdata_tensor[0].clone()
#             _input[variable_order.index(params[0])] = _y
#             _input[variable_order.index(params[1])] = _x
#             posterior = best_gp.posterior(_input.unsqueeze(0))
#             means[i, j] = posterior.mean.item()
#             uncertainties[i, j] = posterior.variance.sqrt().item()
            
#     fig = go.Figure()
#     fig.add_trace(
#         go.Heatmap(z=means, colorscale='Viridis'),
#     )
#     fig.update_layout(
#         title=f'Predicted Stability Slope vs {params[0]} and {params[1]}',
#         yaxis_title=human_names[params[0]],
#         xaxis_title=human_names[params[1]],
#         margin=dict(l=0, r=0, t=30, b=0),  # Remove whitespace around plot
#     )
    
#     fig.write_image(f"{plotfolder}/stability_slope_heatmap_{params[0]}_{params[1]}.png")
#     fig.show()
    


# plot a heatmap of two variables, while fixing the other variables at the best values
# color by the predicted stability slope and uncertainty beside it
print('-'*80)
params_to_plot = [
    ["temperature", "liquid2"],
    ["temperature", "liquid1"],
    ["current_density", "liquid2"],
]
for params in params_to_plot:
    npoints = 50
    yvals = torch.linspace(bounds[variable_order.index(params[0]), 0], bounds[variable_order.index(params[0]), 1], npoints)
    xvals = torch.linspace(bounds[variable_order.index(params[1]), 0], bounds[variable_order.index(params[1]), 1], npoints)
    best_inputs = best_data.iloc[0]
    means = np.zeros((len(yvals), len(xvals)))
    uncertainties = np.zeros((len(yvals), len(xvals)))
    for i, _y in enumerate(yvals):
        for j, _x in enumerate(xvals):
            _input = best_xdata_tensor[0].clone()
            _input[variable_order.index(params[0])] = _y
            _input[variable_order.index(params[1])] = _x
            posterior = best_gp.posterior(_input.unsqueeze(0))
            means[i, j] = posterior.mean.item()
            uncertainties[i, j] = posterior.variance.sqrt().item()
            
    # Create subplot with 2 side-by-side heatmaps
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Predicted Mean', 'Uncertainty (±1σ)'))
    
    # Add mean heatmap
    fig.add_trace(
        go.Heatmap(
            z=means, 
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(x=0.46, len=0.9)
        ),
        row=1, col=1
    )
    
    # Add uncertainty heatmap 
    fig.add_trace(
        go.Heatmap(
            z=uncertainties, 
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(x=1.0, len=0.9)
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f'Predicted Stability Slope vs {params[0]} and {params[1]}',
        margin=dict(l=0, r=0, t=40, b=0),  # Remove whitespace around plot
    )
    
    fig.update_xaxes(title_text=human_names[params[1]], row=1, col=1)
    fig.update_xaxes(title_text=human_names[params[1]], row=1, col=2)
    fig.update_yaxes(title_text=human_names[params[0]], row=1, col=1)
    fig.update_yaxes(title_text=human_names[params[0]], row=1, col=2)
    
    # fig.show()
    fname = f"{plotfolder}/stability_slope_heatmap_uncertainty_sidebyside_{params[0]}_{params[1]}.png"
    fig.write_image(fname)
    print(f"Saved {fname}")


# plot a scatter/heatmap of two variables, while fixing the other variables at the best values
# color by the predicted stability slope, size by uncertainty
print('-'*80)
params_to_plot = [
    ["temperature", "liquid2"],
    ["temperature", "liquid1"],
    ["current_density", "liquid2"],
]
for params in params_to_plot:
    npoints = 50
    yvals = torch.linspace(bounds[variable_order.index(params[0]), 0], bounds[variable_order.index(params[0]), 1], npoints)
    xvals = torch.linspace(bounds[variable_order.index(params[1]), 0], bounds[variable_order.index(params[1]), 1], npoints)
    best_inputs = best_data.iloc[0]
    means = np.zeros((len(yvals), len(xvals)))
    uncertainties = np.zeros((len(yvals), len(xvals)))
    for i, _y in enumerate(yvals):
        for j, _x in enumerate(xvals):
            _input = best_xdata_tensor[0].clone()
            _input[variable_order.index(params[0])] = _y
            _input[variable_order.index(params[1])] = _x
            posterior = best_gp.posterior(_input.unsqueeze(0))
            means[i, j] = posterior.mean.item()
            uncertainties[i, j] = posterior.variance.sqrt().item()
    
    # compute certainty as 1 - uncertainty
    certainties = 1 - uncertainties
    
    # xy grid
    xgrid, ygrid = np.meshgrid(xvals, yvals)
    # make a scatterplot with color as the predicted stability slope
    # and size as the uncertainty
    fig = px.scatter(
        data_frame=pd.DataFrame({
            'x': xgrid.flatten(),
            'y': ygrid.flatten(),
            'stability_slope': means.flatten(),
            'uncertainty': uncertainties.flatten()
        }),
        x='x',
        y='y',
        color='stability_slope',
        size='uncertainty',
        size_max=10  # Reduce default size by 0.5
    )
    fig.update_layout(
        title=f'Predicted Stability Slope vs {params[0]} and {params[1]}',
        xaxis_title=human_names[params[1]],
        yaxis_title=human_names[params[0]],
    )
    # fig.show()
    fname = f"{plotfolder}/stability_slope_heatmap_uncertainty_scatter_{params[0]}_{params[1]}.png"
    fig.write_image(fname)
    print(f"Saved {fname}")



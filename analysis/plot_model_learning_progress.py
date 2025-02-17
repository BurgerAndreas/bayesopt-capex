# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples

import sklearn
# sklearn.decomposition.PCA
# sklearn.preprocessing.StandardScaler, sklearn.preprocessing.Normalizer
import umap

from plotting_helpers import *


# # Is the BayesOpt model learning?
# ### Note: not yet, since this is exploration data, where the model intentionally has high uncertainty
# Plots to illustrate the learning process

data = get_model_progress_over_experiments(recompute=False)

# Plot prediction error
# Should go down over time as the model learns what good parameters are

fig = px.scatter(
    data.reset_index(), 
    x=data.index,
    y='prediction_error',
    color='stability_slope',  # color by actual outcome
    title='Prediction Error vs Experiment Number'
)
fig.update_layout(
    xaxis_title="Experiment Number",
    yaxis_title="Prediction Error",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)  # Remove whitespace around plot
)
fig.write_image(f"{plotfolder}/prediction_error.png")
fig.show()

# Plot relative prediction error
# Should go down over time as the model learns what good parameters are

data["prediction_error_relative"] = data["prediction_error"] / data["stability_slope"]
fig = px.scatter(
    data.reset_index(), 
    x=data.index,
    y='prediction_error_relative',
    color='stability_slope',  # color by actual outcome
    title='Prediction Error vs Experiment Number'
)
fig.update_layout(
    xaxis_title="Experiment Number",
    yaxis_title="Prediction Error",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)  # Remove whitespace around plot
)
fig.write_image(f"{plotfolder}/prediction_error_relative.png")
fig.show()

# plot the uncertainty over the experiments
# Should go down over time as the model learns what good parameters are
# and we are exploiting more instead of exploring

xmin = 15
fig = px.scatter(
    data.reset_index().iloc[xmin:],
    x=data.index[xmin:], # 'experiment',
    y='predicted_std',
    color='stability_slope',  # color by actual outcome
    title='Model Uncertainty vs Experiment Number'
)
fig.update_layout(
    xaxis_title="Experiment Number",
    yaxis_title="Predicted Standard Deviation",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0),  # Remove whitespace around plot
)
fig.show()


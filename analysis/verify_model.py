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
from gpytorch.kernels import RBFKernel
import botorch
from botorch.utils.sampling import draw_sobol_samples
from botorch.cross_validation import gen_loo_cv_folds, batch_cross_validation

from plotting_helpers import *

current_dir = os.path.dirname(os.path.abspath(__file__))
plotfolder = os.path.join(current_dir, "plots")

# set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set default dtype
dtype = torch.float64
torch.set_default_dtype(dtype)

seed = 42
torch.manual_seed(seed)

# load data
data = pd.read_csv("analysis/data.csv", delimiter=";")

# rename the columns
data = data.rename(columns=variable_names)

ycol = "stability_slope"
xcols = [_c for _c in data.columns if _c != ycol]

# reorder the columns based on variable_order
data = data[variable_order + [ycol, "experiment"]]


############################################################################
# # How well can the model predict (fit) the data?
############################################################################
# leave one out cross validation
# LOO-CV computes the average metric on the training data, 
# but evaluated using cross-validation (hold-out one point as test data at a time).

train_X = get_torch_from_df(data, variable_order)
train_Y = torch.tensor(data[ycol].values, dtype=torch.double).unsqueeze(-1)

# Generate LOO-CV folds
cv_folds = gen_loo_cv_folds(train_X=train_X, train_Y=train_Y)

# instantiate and fit model
# the N sets of training data can be fit as N separate GP models 
# with separate hyperparameters in parallel through GPyTorch
cv_results = batch_cross_validation(
    model_cls=SingleTaskGP,
    mll_cls=ExactMarginalLogLikelihood,
    cv_folds=cv_folds,
)

# compute the squared errors for each test point from the prediction and take an average across all cross-validation folds
posterior = cv_results.posterior
cv_error = ((cv_folds.test_Y.squeeze() - posterior.mean.squeeze()) ** 2).mean()
print(f"Cross-validation squared error (absolute): {cv_error : 4.2}")

# LPD is the log probability density to observe the test data under the trained surrogate model.
# higher (0) is better, but high confidence can result in very negative LPD
# For a Gaussian distribution, the probability density function includes a term 1/(σ√(2π)) where σ is the standard deviation
# When the model is very certain (small σ), this term becomes very large
# Taking the log of a very small number (the probability density) results in a very negative number
pred = cv_results.model.posterior(cv_folds.test_X)
lpd = pred.log_prob(cv_folds.test_Y)
print(f"Cross-validation LogProbabilityDensity = {lpd.mean():.4f} (closer to 0 is better)")

# relative squared error
# relative_cv_error = cv_error / ((cv_folds.test_Y.squeeze() - cv_folds.test_Y.squeeze().mean()) ** 2).mean()
relative_cv_error = cv_error / cv_folds.test_Y.mean()
print(f"Cross-validation squared error (relative): {relative_cv_error : 4.2}")

# compute the R2 score
R2 = 1 - cv_error / ((cv_folds.test_Y.squeeze() - cv_folds.test_Y.squeeze().mean()) ** 2).mean()
print(f"R2 score: {R2 : 4.2}")

# Compute Bayesian R² for LOO cross-validation
# Get the predicted means from the posterior
predicted_means = posterior.mean.squeeze()
# Calculate variance of the predicted means
var_mu = torch.var(predicted_means)
# Get the test values (actual observations)
test_values = cv_folds.test_Y.squeeze()
# Calculate residuals
residuals = test_values - predicted_means
# Calculate variance of residuals
var_res = torch.var(residuals)
# Compute Bayesian R²
bayesian_r2_loo = var_mu / (var_mu + var_res)
print(f"Bayesian R² score (LOO-CV): {bayesian_r2_loo:.4f}")



############################################################################
# Noise estimation
############################################################################
print('-'*80)

final_gp = get_my_gp(train_X, train_Y)

# Extract the noise level from the GP model
# The noise parameter is stored in the likelihood of the GP model
noise_level = final_gp.likelihood.noise.item()
print(f"Estimated noise level: {noise_level:.6f}")
print(f"Data signal level (std): {data[ycol].std():.6f}")
print(f"Data signal level (mean): {data[ycol].mean():.6f}")
signal_variance = final_gp.covar_module.lengthscale.mean().item()
# print(f"Model signal variance (lengthscale): {signal_variance:.6f}")

# Get the predicted means from the posterior
predicted_means = posterior.mean.squeeze()
# Calculate variance of the predicted means
var_mu = torch.var(predicted_means)
# Get the test values (actual observations)
test_values = cv_folds.test_Y.squeeze()
# Calculate residuals
residuals = test_values - predicted_means
# Calculate variance of residuals
var_res = torch.var(residuals)
# Compute Bayesian R²
bayesian_r2 = var_mu / (var_mu + var_res)
print(f"Bayesian R² score: {bayesian_r2 : 4.2}")




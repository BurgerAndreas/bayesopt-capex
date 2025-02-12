import numpy as np
import matplotlib.pyplot as plt
import torch 
import pandas as pd


import botorch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP


# most functions you could import from here, but I copied them for convinience
# from run_ML_optimization_activity_stability_Ni_Mo_Andreas import 


#############################################################################################
# Helper functions 
#############################################################################################

# names in the save files -> names in the code
# names in json: 
# "Current density mA/cm2": 22.0,
# "Dep time [s]": 249.0,
# "Dep electrolye T [C]": 33.8,
# "Conc Mo/Ni 10:1 liquid": 0.078,
# "Conc Ni/Mo 10:1 liquid": 0.162,
# "Conc H2SO4": 0.05
# "Integrated stability at 10 [mA/cm2]"
variable_names = {
    "Integrated stability at 10 [mA/cm2]": "stability_slope",
    "Deposition current density [mA/cm2]": "current_density",
    "Current density mA/cm2": "current_density",
    "Dep time [s]": "deposition_time",
    "Dep electrolye T [C]": "temperature",
    "Deposition composition mol / L": "concentrations",
    # "pH_regulation": "pH_regulation",
    "NiSO4": "NiSO4",
    "Na2MoO4": "Na2MoO4",
    "Conc H2SO4": "H2SO4",
    "Conc Ni/Mo 10:1 liquid": "liquid1",
    "Conc Mo/Ni 10:1 liquid": "liquid2",
    # 'integrated_area', 'NiSO4 (mol/L)', 'Na2Mo (mol/L)', 'H2SO4 (mol/L)', 'Dep t (s)', 'Dep I (mA/cm²)', 'Dep T (C)'
    "integrated_area": "stability_slope",
    "NiSO4 (mol/L)": "liquid1",
    "Na2Mo (mol/L)": "liquid2",
    "H2SO4 (mol/L)": "H2SO4",
    "Dep t (s)": "deposition_time",
    "Dep I (mA/cm²)": "current_density",
    "Dep T (C)": "temperature",
}

human_names = {
    "current_density": "Current density [mA/cm2]",
    "deposition_time": "Deposition time [s]",
    "temperature": "Temperature [C]",
    "liquid1": "NiSO4 [mol/L]",
    "liquid2": "Na2MoO4 [mol/L]",
    "H2SO4": "H2SO4 [mol/L]",
    "stability_slope": "Integrated stability at 10 [mA/cm2]",
}

# tensors of X, Y, bounds, buckets, etc. need to be stacked in this order
variable_order = [
    "current_density",
    "deposition_time",
    "temperature",
    "liquid1", 
    "liquid2",
    "H2SO4",
    # "NiSO4", "Na2MoO4",
]

# min and max values for the variables
parameter_bounds = {
    "current_density": [1, 200],  # in mA/cm2
    "deposition_time": [60, 600],  # in seconds
    "pH_regulation": [0, 1.5],  # in ml
    "temperature": [30, 70],  # in degrees C
    "liquid1": [0.002, 0.4],  # in mol/L
    "liquid2": [0.002, 0.4],  # in mol/L
    "NiSO4": [0.002, 0.4],  # in mol/L
    "Na2MoO4": [0.002, 0.4],  # in mol/L
    "H2SO4": [0, 0.1],  # in mol/L
}


# how precise the variables can be changed
parameter_granularity = {
    "current_density": -1,  # in mA/cm2
    "deposition_time": -1,  # in seconds
    "pH_regulation": 0.075,  # in ml
    "temperature": 1,  # in degrees C
    "NiSO4": 0.002,  # 1/200 * 0.4 in mol/L
    "Na2MoO4": 0.002,  # 1/200 * 0.4 in mol/L
    "H2SO4": 0.005,  # 1/200 * 1 in mol/L
}

# buckets are the possible values for the variables
buckets = {
    "current_density": None,  # in mA/cm2
    "deposition_time": None,  # in seconds
    # "pH_regulation": torch.arange(0, 1.5, 0.075),  # in ml
    "temperature": torch.arange(30, 70, 1),  # in degrees C
    # "NiSO4": torch.arange(0, 0.4, 0.002),  # 1/200 * 0.8 in mol/L
    # "Na2MoO4": torch.arange(0, 0.4, 0.002),  # 1/200 * 0.8 in mol/L
    "liquid1": torch.arange(0.002, 0.4, 0.002),  # 1/200 * 0.4 in mol/L
    "liquid2": torch.arange(0.002, 0.4, 0.002),  # 1/200 * 0.4 in mol/L
    "H2SO4": torch.arange(0, 0.1, 0.05),  # 1/20 * 0.1 in mol/L
}

# the three input compounds are solved in water at a certain concentration
compound_concentrations = {
    "NiSO4": 0.4,  # in mol/L
    "Na2MoO4": 0.4,  # in mol/L
    "H2SO4": 1,  # in mol/L
}
# list is easier to handle
stock_concentrations = [0.4, 0.4, 1]

# Define the bounds for the variables
bounds = torch.tensor([parameter_bounds[v] for v in variable_order], dtype=torch.double)

inequality_constraints = [
    # Constraint 1: more of one liquid reduces the concentration of the other liquids
    # Conc_stock_liquid_1 = 0.4 
    # Conc_stock_liquid_2 = 0.4 
    # Conc_stock_H2SO4 = 1 
    # Sum of Conc_liquid_1 / Conc_stock_liquid_1 + Conc_liquid_2 / Conc_stock_liquid_2 + Conc_H2SO4 / Conc_stock_H2SO4 < 1
    # Sum of Conc_liquid_1 / 0.4 + Conc_liquid_2 / 0.4 + Conc_H2SO4 / 1 < 1
    # Conc_liquid_1 * 2.5 + Conc_liquid_2 * 2.5 + Conc_H2SO4 * 1 < 1
    # list of tuples (indices, coefficients, rhs)
    # \sum_i (X[indices[i]] * coefficients[i]) >= rhs
    (
        # indices of the variables we want to constrain
        torch.tensor(
            [variable_order.index("liquid1"), variable_order.index("liquid2"), variable_order.index("H2SO4")],
            dtype=torch.long,
        ),
        # coefficients of the linear combination (weighted sum)
        # Conc_liquid_1 * 2.5 + Conc_liquid_2 * 2.5 + Conc_H2SO4 * 1 <= 1
        # default is >=, so we need to flip the sign of the coefficients
        -1 * torch.tensor([1.0/0.4, 1.0/0.4, 1.0], dtype=torch.double),
        # bigger or equal to
        -1.0,
    ),
    # Constraint 2: Use a minimum amount of liquid 
    # liquid_1 + liquid_2 >= 0.05
    (
        torch.tensor(
            [variable_order.index("liquid1"), variable_order.index("liquid2")],
            dtype=torch.long,
        ),
        torch.tensor([1.0, 1.0], dtype=torch.double),
        0.05,
    ),
]

#############################################################################################
# Data handling
#############################################################################################

ycol = "stability_slope"
def get_data_from_file():
    data = pd.read_csv("data.csv", delimiter=";")

    # rename the columns
    data = data.rename(columns=variable_names)

    xcols = [_c for _c in data.columns if _c != ycol]

    # reorder the columns based on variable_order
    data = data[variable_order + [ycol, "experiment"]]
    return data

def get_torch_from_df(_df, column_names):
    cols_values = []
    for v in column_names:
        cols_values.append(_df[v].values)
    _data = torch.tensor(np.array(cols_values).T, dtype=torch.double)
    assert _data.shape[1] == len(column_names), f"Expected {len(column_names)} columns, got {_data.shape[1]}"
    assert _data.shape[0] == _df.shape[0], f"Expected {_df.shape[0]} rows, got {_data.shape[0]}"
    return _data

def get_my_gp(_x, _y):
    # Define the GP model
    return SingleTaskGP(
        train_X=_x,
        train_Y=_y,
        outcome_transform=botorch.models.transforms.Standardize(m=1),
        input_transform=botorch.models.transforms.Normalize(d=6, bounds=bounds.T),
    )

def get_model_progress_over_experiments(recompute=False):
    # we didn't save the model during the optimization process
    # so we need to load the data and re-run the optimization at each iteration
    # only redo this if you have new data
    # only set recompute to True if you have new data, 
    # set back to False after running the optimization once

    data_with_predictions_filename = "data_with_predictions.csv"
    load_from_file = not recompute 

    if load_from_file:
        data = pd.read_parquet(data_with_predictions_filename)
        print(f"Loaded from {data_with_predictions_filename}")
        
    else:
        # add new columns to the data
        data["predicted_mean"] = None
        data["predicted_std"] = None

        xdatatensor = get_torch_from_df(data, variable_order)
        ydatatensor = torch.tensor(data[ycol].values, dtype=torch.double).unsqueeze(-1)

        for i in range(1, len(data)):
            
            # get data up to i-1
            # as numpy of shape [n_experiments, n_variables]
            xdata = xdatatensor[:i]
            ydata = ydatatensor[:i]
            
            # Refit the GP model with the new experimental data
            gp = get_my_gp(xdata, ydata)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll = fit_gpytorch_mll(mll)
            
            # get the i-th experiment
            suggested_experiment = xdatatensor[i]
            print(i, suggested_experiment)
            
            posterior = gp.posterior(suggested_experiment.unsqueeze(0))
            predicted_mean = posterior.mean  # Predicted y-value (mean)
            predicted_std = posterior.variance.sqrt()  # Prediction uncertainty (std)
            
            # update the data
            data.loc[i, "predicted_mean"] = predicted_mean
            data.loc[i, "predicted_std"] = predicted_std

        # compute the prediction error
        data["prediction_error"] = data["predicted_mean"] - data[ycol]

        # Convert tensor data to numpy/float before plotting
        data = data.copy()
        data['predicted_mean'] = data['predicted_mean'].apply(lambda x: float(x) if x is not None else None)
        data['predicted_std'] = data['predicted_std'].apply(lambda x: float(x) if x is not None else None)
        data['prediction_error'] = data['prediction_error'].apply(lambda x: float(x) if x is not None else None)
        
        data.to_parquet(data_with_predictions_filename)
        print(f"Saved to {data_with_predictions_filename}")
        
    return data
    
plotfolder = "plots"

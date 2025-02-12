import sys
import os
import threading
import torch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import botorch.acquisition
try:
    from botorch.acquisition import qLogExpectedImprovement
    acqfcnt = qLogExpectedImprovement
    acqfcnt_str = "qLogExpectedImprovement"
except:
    from botorch.acquisition import qExpectedImprovement
    acqfcnt = qExpectedImprovement
    acqfcnt_str = "qExpectedImprovement"
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition import qMaxValueEntropy
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
analysis_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Live_data_analysis")
)
sys.path.append(analysis_path)
# Now you can import from the parent directory
from Catbot_control_master import *
from utils import *
from experimental_protocols import *
from experiment_class import *
from Live_data_analysis import live_data_analysis_after_testing as data_analysis


# REAL_ROBOT = 

Robot_test = CatBot(
    serialcomm_temp="COM4", serialcomm_liquid="COM6"
)  # Initialize a catbot with two different serialcomms

time.sleep(15)
# Testing experiment that you want to run
EC_data_path = (
    r"C:\Users\Catbot-adm\Desktop\EC_data_CatBot\Ni_Mo_optimization\Data_dicts"
)
Robot_test.evacuate_all_tubings(evacuation_volume=4)

ML_data_log_path_save = "C:\\Users\\Catbot-adm\\Desktop\\CatBot\\Python\\Electrochemical_data\\Electrochemical_data_second_phase\\Data_dicts\\ML"

ECSA_dict_path = os.path.join(EC_data_path, "CV_ECSA_dict_all_data.json")
CP_dict_name_path = os.path.join(EC_data_path, "CP_datadict_all_data.json")
CV_cycling_stability_dict_path = os.path.join(
    EC_data_path, "CV_cycling_stability_dict_all_data.json"
)
EIS_dict_path = os.path.join(EC_data_path, "EIS_dict_all_data.json")
LSV_dict_path = os.path.join(EC_data_path, "LSV_dict_all_data.json")

shift = 0 # The shift in reference electrode. Remember we may use a Ag/AgCl reference

nickel_calibration_test = nickel_calibration_KOH(shift=-shift)
testing_experiment, testing_protocol_name = Ni_Mo_optimization_testing_protocol(shift=-shift)
current_density_mA_cm2_stability = 10 # The current density at which we evaluate the CV stability cycling 
output_data_folder = r"C:\Users\Catbot-adm\Desktop\EC_data_CatBot\Ni_Mo_optimization"

# Example experiment
experiment = Experiment(
    experimental_params={
        "Temperature_deposition [C]": 35,
        "Temperature_testing [C]": 80,
        "Testing liquid KOH [w %]": 30,
        "Deposition composition": {"NiSO4": 0.4, "Na2Mo" : 0, "H2SO4": 0},
        "Roll while depositing": True,
        "Testing protocol": {
            "testing protocol name": testing_protocol_name,
            "protocol": testing_experiment,
        },
        "Deposition time [s]": 20,
        "Deposition current density [mA/cm2]": 25,
        "Wire type": "Ni 99.8 %",
        "Filename testing data": "",
        "Filename deposition data": "",
        "Filename temperature data": "",
        "Filename folder": "",
        "General comments": "ML optimization run Ni-Mo starting January 2025. Goal optimize activity and stability using integrated area energy metric",
        "Clean after testing": True,
        "Maintain KOH after testing": False,  # This is to do retained electrolyte experiments
        "Optimize using ML": False,
        "HCl dipping time [s]": 900,
        "HCl cleaning concentration [mol / L]": 3,
        "KOH filling volume [ml]": 10.9,
        "Deposition filling volume [ml]": 15,
        "Experiment name": "",
        "KOH batch": "Batch fabricated 05.01.2025 Pre electrolyzed 72 h 30 wt %",
        "Repeat experiment n cycles": 1,  # How many times you want to repeat the experiment
        "Cleaning waiting time testing [s]": 60,
        "Cleaning waiting time deposition [s]": 60,
        "Cleaning cycles testing chamber": 2,
        "Cleaning cycles deposition chamber": 2,
        "Deposition batch": "Electrolytes are NiSO4 0.4 M + 0.4 M NaCitr + 0.04M Na2Mo pH = 9 [pump 6] and Electrolytes are Na2Mo 0.4 M + 0.4 M NaCitr + 0.04 M NiSO4 pH = 9 [pump 7] and 1 M H2SO4 [pump 4]",
        "HCl concentration [mol / L]": 3,
    }
)


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
}

# tensors of X, Y, bounds, buckets, etc. need to be stacked in this order
variable_order = [
    "current_density",
    "deposition_time",
    "temperature",
    "liquid1", # 3
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


def individiual_to_mixture_concentrations(c_NiSO4, c_Na2MoO4):
    """Convert individual concentrations to mixture concentrations.
    before: stock concentrations
    - NiSO4 at 0.8 mol/L, Na2MoO4 at 0.8 mol/L
    after: mixture concentrations
    - liquid 1 with 1:10 ratio 0.4 NiSO4:0.04 Na2MoO4,
    - liquid 2 with 10:1 ratio 0.4 Na2MoO4:0.04 NiSO4
    - liquid 3 with 1 M H2SO4
    - liquid 4 H2O milliq
    """
    # first ensure that the concentrations are within the bounds of the new liquids
    if c_NiSO4 > 0.2 or c_Na2MoO4 > 0.2:
        raise ValueError(
            f"Concentrations are too high for the new liquids: {c_NiSO4}, {c_Na2MoO4}"
        )
    if c_NiSO4 < 0.04 or c_Na2MoO4 < 0.04:
        raise ValueError(
            f"Concentrations are too low for the new liquids: {c_NiSO4}, {c_Na2MoO4}"
        )
    
    # this is a small system of linear equations
    # c_NiSO4 = (c_liquid1 * 0.4) + (c_liquid2 * 0.04)
    # c_Na2MoO4 = (c_liquid1 * 0.04) + (c_liquid2 * 0.4)
    # solve for c_liquid1 and c_liquid2 with numpy
    A = np.array([[0.4, 0.04], [0.04, 0.4]])
    b = np.array([c_NiSO4, c_Na2MoO4])
    return np.linalg.solve(A, b)


def clean_data(xdata, ydata):
    """Combine the xdata and ydata into a single dataframe.
    xdata: pandas dataframe with the input parameters
    ydata: pandas dataframe with the goal parameters
    returns: pandas dataframe with the combined data, and the number of compounds
    """
    xdata = xdata.transpose()
    ydata = ydata.transpose()

    # combine xdata and ydata
    # set index to experiment_name
    xdata["experiment_name"] = xdata.index
    ydata["experiment_name"] = ydata.index
    # match xdata and ydata based on experiment_name (index column)
    data = xdata.merge(ydata, on="experiment_name")

    return data


def df_to_numpy(data):
    """Convert a pandas dataframe to a torch tensor.
    data: pandas dataframe
    """
    data = data.rename(columns=variable_names)
    print(f"Data columns:\n {data.columns}")
    x = data[variable_order].values
    print(x,"Training data")
    discarded_columns = [col for col in data.columns if col not in variable_order]
    print(f"Discarded columns:\n {discarded_columns}")
    y = data[["stability_slope"]].values
    return x, y


#############################################################################################
# Load data 
#############################################################################################

# set random seeds for numpy and torch and scipy
np.random.seed(0)
torch.manual_seed(0)

save_path_ML_data = "C:/Users/Catbot-adm/Desktop/EC_data_CatBot/Ni_Mo_optimization/ML"


def get_data_andreas():
    # xdata = pd.read_json("Input_parameters_database.json")
    xdata = pd.read_json(os.path.join(
        save_path_ML_data, "Input_parameters_database.json"
    ))
    # ydata = pd.read_json("Goal_parameters_database.json")
    ydata = pd.read_json(os.path.join(
        save_path_ML_data, "Goal_parameters_database.json"
    ))
    
    # print(f"keys in xdata: {xdata[xdata.keys()[0]].keys()}")
    # print(f"keys in ydata: {ydata[ydata.keys()[0]].keys()}")

    # combine the two
    data = clean_data(xdata, ydata)

    x, y = df_to_numpy(data)

    return np.vstack(x).astype(float), y


train_X, train_Y = get_data_andreas()
train_X = torch.tensor(train_X, dtype=torch.double)
train_Y = torch.tensor(train_Y, dtype=torch.double)
train_Y = -train_Y # We are going to maxmimize the negative integrated area, i.e make it smaller 
######################################################################
# Definition of the bounds, constraints, and BayesOpt model
######################################################################
print(train_Y)


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
    # default is >=, so we need to flip the sign of the coefficients
    (
        # indices of the variables we want to constrain
        torch.tensor(
            [variable_order.index("liquid1"), variable_order.index("liquid2"), variable_order.index("H2SO4")],
            dtype=torch.long,
        ),
        # coefficients of the linear combination (weighted sum)
        # Conc_liquid_1 * 2.5 + Conc_liquid_2 * 2.5 + Conc_H2SO4 * 1 <= 1
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

def check_inequality(suggested_experiment):
    # check if the constraints are satisfied
    for indices, coefficients, rhs in inequality_constraints:
        # calculate the linear combination
        # \sum_i (X[indices[i]] * coefficients[i])
        linear_combination = torch.sum(suggested_experiment[indices] * coefficients)
        formula = " + ".join(
            [f"{coeff:.3f} * {suggested_experiment[idx]:.3f}" for idx, coeff in zip(indices, coefficients)]
        )
        # check if the constraint is satisfied
        print(formula, f"= {linear_combination:.3f} >= {rhs:.3f} (satisfied: {linear_combination >= rhs})")
        assert(linear_combination * -1 <= 1), f"Cosntraint not satisfied: {linear_combination * -1} <= 1"


def get_my_gp(_x, _y):
    # Define the GP model
    return SingleTaskGP(
        train_X=_x,
        train_Y=_y,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=6, bounds=bounds.T),
    )


######################################################################
# Train BayesOpt model on initial data
######################################################################

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
print(np.shape(train_X), np.shape(train_Y))
gp = get_my_gp(train_X, train_Y) # This actually normalizes the input parameters 

candidate_set = torch.rand(10000, train_X.size(1))  # Large random set
candidate_set = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * candidate_set # The possible candidates we investigate

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
mll = fit_gpytorch_mll(mll)

"""Possible acquisition functions
from more exploitation to more exploration
qNoisyExpectedImprovement
qUpperConfidenceBound | beta=2
qMaxValueEntropySearch
"""
acqfcnt = qMaxValueEntropy
acqfcnt_str = "qMaxValueEntropy"

######################################################################
# Start the optimization loop with real experiments
######################################################################
max_iterations = 4  # TODO: set this to whatever you want
experiments_per_iteration = 1 # TODO: set this to whatever you want

for iteration in range(max_iterations):
    # Define the acquisition function
    
    # Usec to be max (That is maximize integrated area) instead we have now switched to min 
    #qEI = acqfcnt(model=gp, best_f=train_Y.max()) 
    maxVE = acqfcnt(model=gp, 
                    maximize = True, 
                    candidate_set = candidate_set) # Max value entropy search 
    # Optimize the acquisition function
    # acq_values = new_Ys
    new_Xs, acq_values = optimize_acqf(
        acq_function=maxVE,
        bounds=bounds.T,
        q=experiments_per_iteration,  # number of candidates
        inequality_constraints=inequality_constraints,
        num_restarts=10,
        raw_samples=50,
    )

    suggested_experiments = new_Xs
    method = acqfcnt_str

    if len(suggested_experiments.shape) == 1:
        suggested_experiments = suggested_experiments.unsqueeze(0)
    # loop over the suggested experiments
    for expnum, new_X in enumerate(suggested_experiments):
        # round parameters to nearest bucket
        for i, v in enumerate(new_X):
            possible_values = buckets[variable_order[i]]
            # only round if the bucket is not -1 (continuous variable)
            # print(f"variable_order[i]", variable_order[i])
            if possible_values is not None:
                new_X[i] = possible_values[torch.argmin(torch.abs(possible_values - v))]
                # print(f"v={v}")
                # print(f"v={new_X[i]} (bucketed)")

        suggested_experiment = new_X
        print(suggested_experiment)
        print(suggested_experiments.shape)
        # get the models prediction for the suggested experiment
        posterior = gp.posterior(suggested_experiment.unsqueeze(0))
        predicted_mean = posterior.mean  # Predicted y-value (mean)
        predicted_std = posterior.variance.sqrt()  # Prediction uncertainty (std)
        
        print("-" * 80)
        print(f"New suggested experiment (iteration={iteration}, experiment={expnum}):")
        for _, _v in enumerate(suggested_experiment):
            print(f"{variable_order[_]}: {_v:.3f}")
        print(f'Acquisition value: {acq_values.item()}')
        print(f"Predicted y (slope): {predicted_mean.item()}")
        print(f"Prediction uncertainty: {predicted_std.item()}")
        print(f"Constraint:")
        check_inequality(suggested_experiment)

        # Saves the suggested experiments to ML dictionary
        # Error comes because the suggested experiments have a too high concentration
        # Tomorrow, we implement saving suggested experiments to ML log
        # Then we try to expand the boundary such that we actually have our specified boundaries,
        # and not the boundaries that come from the x-values

        save_suggested_experiments_to_ML_log(
            suggested_experiment, save_path_ML_data, method=method,
            iteration=iteration, experiment=expnum,
            acquisition_value=acq_values.item(), # expected improvement over previous best experiment
            predicted_y=predicted_mean, # predicted y value (integrated slope) of the model
            prediction_uncertainty=predicted_std, # uncertainty of the model about y value
        )
        
        ############################################################################
        # TODO: Paolo's code. check that variable names / order is correct

        # Our variables are [I (mA/cm2), dep time [s],temp [C], NiSO4 0.4 [mol/L], Mo [mol / L], pH liquid 0-1.5 ml]
        current_density_mA_cm2_ML_exp_i = round(float(suggested_experiment[0]), 1)
        deposition_time_s_ML_exp_i = round(float(suggested_experiment[1]), 1)
        temp_dep_C_ML_exp_i = round(float(suggested_experiment[2]), 1)

        deposition_composition_mol_l_Ni = round(float(suggested_experiment[3]), 4)
        deposition_composition_mol_l_Mo = round(float(suggested_experiment[4]), 4)
        deposition_composition_mol_l_pH_reg = round(float(suggested_experiment[5]), 4)

        # Setting parameters for electrochemical experiments
        experiment.dep_current_density_mA_cm2 = current_density_mA_cm2_ML_exp_i
        experiment.deposition_time_s = deposition_time_s_ML_exp_i
        experiment.dep_temperature = temp_dep_C_ML_exp_i
        # Setting the concentratios of the NiSO4, Na2Mo and pH regulation liquid
        experiment.deposiotion_composition_mol_l["NiSO4"] = deposition_composition_mol_l_Ni
        experiment.deposiotion_composition_mol_l["Na2Mo"] = deposition_composition_mol_l_Mo
        experiment.deposiotion_composition_mol_l["H2SO4"] = deposition_composition_mol_l_pH_reg

        print("Experiment parameters: (density, time, liquid1, T)")
        print(experiment.dep_current_density_mA_cm2, current_density_mA_cm2_ML_exp_i)
        print(experiment.deposition_time_s, deposition_time_s_ML_exp_i)
        print(
            experiment.deposiotion_composition_mol_l["NiSO4"],
            deposition_composition_mol_l_Ni,
        )
        print(experiment.dep_temperature, temp_dep_C_ML_exp_i)

        # Delete hastags when ready
        Robot_test.run_complete_experiment(experiment=experiment, 
                empty_after_deposition=True, 
                keep_wire_stationary=False, 
            evacuate_chambers_before_starting=True, 
            output_data_folder=output_data_folder, 
            filename_keywords = "Ni_Mo_opt", 
            reference_electrode_shift = -shift * 1000, 
            nickel_calibration_exp=nickel_calibration_test
            )

        
        dataset_path = os.path.join(
            output_data_folder, experiment.filename_testing_data
        )

        try:
            
            IR_correction,T_dep_electrolyte_C, integrated_stability = data_analysis.analyze_data_after_testing(datafile_path=dataset_path, 
                            ECSA_dict_path=ECSA_dict_path, 
                            CV_cycling_stability_dict_path=CV_cycling_stability_dict_path, 
                            EIS_dict_path=EIS_dict_path, 
                            LSV_dict_path=LSV_dict_path, 
                            CP_dict_name_path=CP_dict_name_path,
                            override_previous_data=True,
                            save_plotted_data=True, 
                            use_integrated_stability = True,
                            current_density_mA_cm2_stability = current_density_mA_cm2_stability,
                            CV_stability_cycling_scan_rate=50, 
                            shift = shift * 1000) # Get the THE drift in mV
        except:
             IR_correction,T_dep_electrolyte_C, integrated_stability = 0, 0, 0 # In case of failed experiment, just continue the experiment so that we dont get this annoying feature 
        
        # same order as variable_order
        
        
        # Updated these 
        x_data_experiment = {"Current density mA/cm2": current_density_mA_cm2_ML_exp_i, 
                             "Dep time [s]" : deposition_time_s_ML_exp_i, 
                             "Dep electrolye T [C]" : T_dep_electrolyte_C,
                             "Conc Mo/Ni 10:1 liquid" : deposition_composition_mol_l_Mo,
                             "Conc Ni/Mo 10:1 liquid" : deposition_composition_mol_l_Ni,
                             "Conc H2SO4" : deposition_composition_mol_l_pH_reg
        }
        
        y_data_experiment = {f"Integrated stability at {current_density_mA_cm2_stability} [mA/cm2]" : integrated_stability}

        # TODO: fix path / filenames
        save_parameter_sets(
            input_parameter_set_path=os.path.join(
                save_path_ML_data, "Input_parameters_database.json"
            ),
            goal_parameter_set_path=os.path.join(
                save_path_ML_data, "Goal_parameters_database.json"
            ),
            input_params_x=[x_data_experiment],
            goal_params_y=[y_data_experiment],
            experiment_names=[experiment.experiment_name.split(".csv")[0]],
            IR_correction=IR_correction,
        )

        # Paolo's code ends here
        ############################################################################

        # Update the training data
        # usually experimental parameters deviate from the intended values
        
        # (num_samples, num_variables)
        x_data_experiment = [
            current_density_mA_cm2_ML_exp_i,
            deposition_time_s_ML_exp_i,
            T_dep_electrolyte_C,
            deposition_composition_mol_l_Ni, # liquid1
            deposition_composition_mol_l_Mo, # liquid2
            deposition_composition_mol_l_pH_reg, # H2SO4
        ]

        x_data_experiment = torch.tensor(x_data_experiment, dtype=torch.double)
        if len(x_data_experiment.shape) == 1:
            x_data_experiment = x_data_experiment.unsqueeze(0)
        train_X = torch.cat([train_X, x_data_experiment], dim=0)

        # (num_samples, 1)
        new_Y = torch.tensor([-integrated_stability])
        if len(new_Y.shape) == 1:
            new_Y = new_Y.unsqueeze(0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        

    # Refit the GP model with the new experimental data
    gp = get_my_gp(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll = fit_gpytorch_mll(mll)

best_idx = torch.argmax(train_Y)
print(f"Best experimental settings:")
for i, v in enumerate(train_X[best_idx]):
    print(f"{variable_order[i]}: {v:.3f}")
print(f"Best integrated stability: {train_Y[best_idx].item()}")

Robot_test.pump_KOH_into_testing_chamber(amount_ml=10)
Robot_test.set_temperature_both_chambers(
    filename="bs.json", temperature_dep_electrolyte=30, temperature_KOH=30
)

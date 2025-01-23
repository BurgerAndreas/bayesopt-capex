import pandas as pd
import numpy as np


variable_names = {
    "Stability slope [mV/scan]": "stability_slope",
    "Deposition current density [mA/cm2]": "current_density",
    "Deposition time [s]": "deposition_time",
    "Temperature_deposition [C] realized": "temperature",
    "Deposition composition mol / L": "concentrations",
    # TODO: how is pH_regulation called in the database?
    "pH_regulation": "pH_regulation",
    "NiSO4": "NiSO4",
    "Na2MoO4": "Na2MoO4",
    "H2SO4": "H2SO4",
    # TODO: how are the new liquids called in the database?
    "liquid1": "liquid1",
    "liquid2": "liquid2",
}

variable_order = [
    "current_density", "deposition_time", "pH_regulation",
    "temperature", 
    "liquid1", "liquid2",
    #"NiSO4", "Na2MoO4",
]

parameter_bounds = {
    "current_density": [1, 100],  # in mA/cm2
    "deposition_time": [30, 360],  # in seconds
    "pH_regulation": [0, 0.4],  # in ml
    "temperature": [30, 80],  # in degrees C
    "liquid1": [0, 1],  # in mol/L
    "liquid2": [0, 1],  # in mol/L
    "NiSO4": [0, 0.8],  # in mol/L
    "Na2MoO4": [0, 0.8],  # in mol/L
    "H2SO4": [0, 3],  # in mol/L
}

parameter_granularity = {
    "current_density": -1,  # in mA/cm2
    "deposition_time": -1,  # in seconds
    "pH_regulation": 0.0015,  # in ml
    "temperature": 1,  # in degrees C
    "NiSO4": 0.004,  # 1/200 * 0.8 in mol/L
    "Na2MoO4": 0.004,  # 1/200 * 0.8 in mol/L
    "H2SO4": 0.015,  # 1/200 * 3 in mol/L
}
buckets = {
    "current_density": [],  # in mA/cm2
    "deposition_time": [],  # in seconds
    "pH_regulation": np.arange(0, 0.4, 0.0015),  # in ml
    "temperature": np.arange(30, 80, 1),  # in degrees C
    "NiSO4": np.arange(0, 0.8, 0.004),  # 1/200 * 0.8 in mol/L
    "Na2MoO4": np.arange(0, 0.8, 0.004),  # 1/200 * 0.8 in mol/L
    "H2SO4": np.arange(0, 3, 0.015),  # 1/200 * 3 in mol/L
}

# the three input compounds are solved in water at a certain concentration
compound_concentrations = {
    "NiSO4": 0.8,  # in mol/L
    "Na2MoO4": 0.8,  # in mol/L
    "H2SO4": 3,  # in mol/L 
}
# list is easier to handle
stock_concentrations = [0.8, 0.8, 3]

def individiual_to_mixture_concentrations(c_NiSO4, c_Na2MoO4):
    """Convert individual concentrations to mixture concentrations.
    before: stock concentrations
    - NiSO4 at 0.8 mol/L, Na2MoO4 at 0.8 mol/L   
    after: mixture concentrations
    - liquid 1 with 1:10 ratio 0.4 NiSO4:0.04 Na2MoO4, 
    - liquid 2 with 10:1 ratio 0.04 Na2MoO4:0.4 NiSO4
    """
    # first ensure that the concentrations are within the bounds of the new liquids
    if c_NiSO4 > 0.4 or c_Na2MoO4 > 0.4:
        raise ValueError(f"Concentrations are too high for the new liquids: {c_NiSO4}, {c_Na2MoO4}")
    if c_NiSO4 < 0.04 or c_Na2MoO4 < 0.04:
        raise ValueError(f"Concentrations are too low for the new liquids: {c_NiSO4}, {c_Na2MoO4}")
    
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
    
    # numer of compounds
    num_compounds = len(xdata["Deposition composition mol / L"].iloc[0])
    
    # "Deposition composition mol / L" is a dictionary
    # add a column for each key in the dictionary
    for key in xdata["Deposition composition mol / L"].iloc[0]:
        print(f"key: {key}")
        print(f"value: {xdata['Deposition composition mol / L'].iloc[0][key]}")
        xdata[key] = xdata["Deposition composition mol / L"].apply(lambda x: x[key])
    xdata.head()

    # combine xdata and ydata
    # set index to experiment_name
    xdata["experiment_name"] = xdata.index
    ydata["experiment_name"] = ydata.index
    # match xdata and ydata based on experiment_name (index column)
    data = xdata.merge(ydata, on="experiment_name")

    return data, num_compounds

def df_to_torch(data):
    """Convert a pandas dataframe to a torch tensor.
    data: pandas dataframe
    """
    data = data.rename(columns=variable_names)
    print(f"Data columns:\n {data.columns}")
    x = data[variable_order].values
    # add compounds if they are in the data
    if "H2SO4" in data.columns:
        x = np.hstack([x, data[["H2SO4"]].values])
    discarded_columns = [col for col in data.columns if col not in variable_order]
    print(f"Discarded columns:\n {discarded_columns}")
    y = data[["stability_slope"]].values
    return x, y

# example usage
if __name__ == "__main__":
    
    # test: should be 1 and 0
    print(individiual_to_mixture_concentrations(0.4, 0.04))
    # should be 0 and 1
    print(individiual_to_mixture_concentrations(0.04, 0.4))
    
    print("Number of possible parameter values:")
    for b, v in buckets.items():
        print(b, len(v))
    
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
    data["liquid1"], data["liquid2"] = zip(*data.apply(lambda row: individiual_to_mixture_concentrations(row["NiSO4"], row["Na2MoO4"]), axis=1))
    
    x, y = df_to_torch(data)
    
    # print(x)
    # print(y)
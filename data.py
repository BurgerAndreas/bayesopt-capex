import pandas as pd
import numpy as np


variable_names = {
    "Stability slope [mV/scan]": "stability_slope",
    "Deposition current density [mA/cm2]": "current_density",
    "Deposition time [s]": "deposition_time",
    "Temperature_deposition [C] realized": "temperature",
    "Deposition composition mol / L": "concentrations",
    # TODO: what is pH_regulation?
    "pH_regulation": "pH_regulation",
    "NiSO4": "NiSO4",
    "Na2MoO4": "Na2MoO4",
    "H2SO4": "H2SO4",
}

variable_order = [
    "current_density", "deposition_time", "pH_regulation",
    "temperature", "NiSO4",
]

parameter_bounds = {
    "current_density": [1, 100],  # in mA/cm2
    "deposition_time": [30, 360],  # in seconds
    "pH_regulation": [0, 0.4],  # in ml
    "temperature": [30, 80],  # in degrees C
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

# the three input compounds are solved in water at a certain concentration
compound_concentrations = {
    "NiSO4": 0.8,  # in mol/L
    "Na2MoO4": 0.8,  # in mol/L
    "H2SO4": 3,  # in mol/L 
}
# list is easier to handle
stock_concentrations = [0.8, 0.8, 3]



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
    if "Na2MoO4" in data.columns:
        x = np.hstack([x, data[["Na2MoO4"]].values])
    if "H2SO4" in data.columns:
        x = np.hstack([x, data[["H2SO4"]].values])
    discarded_columns = [col for col in data.columns if col not in variable_order]
    print(f"Discarded columns:\n {discarded_columns}")
    y = data[["stability_slope"]].values
    return x, y

# example usage
if __name__ == "__main__":
    xdata = pd.read_json("input_parameters_database.json")
    ydata = pd.read_json("goal_parameters_database.json")
    
    
    # combine the two
    data, num_compounds = clean_data(xdata, ydata)
    
    # data is missing pH_regulation, so add 0 everywhere
    if "pH_regulation" not in data.columns:
        data["pH_regulation"] = 0.0
    
    x, y = df_to_torch(data)
    
    print(x)
    print(y)
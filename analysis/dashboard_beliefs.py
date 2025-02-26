"""
Dashboard for Visualizing Model Predictions

This dashboard loads the CSV data, computes the final GP model (best state),
and lets you:
- In Section 1, select a variable to sweep via a dropdown and adjust the other
variable values via sliders. The predicted mean (with ±1σ uncertainty) is shown
in a line plot positioned to the right of the sliders.
- In Section 2, select two variables for a heatmap via dropdowns and adjust the
remaining variable values via sliders (displayed on the left); the right side shows
two heatmaps (stacked vertically): one with the predicted stability slope and one
with its uncertainty.
- In Section 3, we plot good regions using dimensional reduction techniques (work in progress)
  
Make sure that your helper functions and globals are available by importing them
from your "plotting_helpers" module.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, ALL  # Pattern matching for dynamic components.
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np
import pandas as pd

# Import helper functions and globals.
# They are defined in your plotting_helpers file:
#   - get_torch_from_df: converts a pandas DataFrame to a torch tensor following variable_order.
#   - get_my_gp: returns the final GP (posterior) model.
#   - variable_names: a mapping from original names to new names.
#   - variable_order: the list of input variable names in a prescribed order.
#   - human_names: a mapping of variable names to human‐readable labels.
#   - bounds: an array (or list) with each row giving the [min, max] for the variable.
from plotting_helpers import (
    get_torch_from_df,
    get_my_gp,
    variable_names,
    variable_order,
    human_names,
    bounds
)

# ------------------------ Data and Model Setup ------------------------------

# Load CSV data (change the path if necessary)
data = pd.read_csv("analysis/data4.csv", delimiter=";")
data = data.rename(columns=variable_names)
ycol = "stability_slope"
# Reorder columns according to variable_order (plus y and 'experiment' if available)
data = data[variable_order + [ycol, "experiment"]]

# Convert data to torch tensors for the GP model fitting.
xdatatensor = get_torch_from_df(data, variable_order)
ydatatensor = torch.tensor(data[ycol].values, dtype=torch.double).unsqueeze(-1)
best_gp = get_my_gp(xdatatensor, ydatatensor)

# Choose the best data point (e.g., the one with the lowest stability_slope)
best_data = data.sort_values(by=ycol, ascending=True)
best_xdata_tensor = get_torch_from_df(best_data, variable_order)
# Use the first row (i.e., the best) as the default base input vector.
base_input_default = best_xdata_tensor[0].clone()

# -------------------------- Initialize the App ------------------------------

app = dash.Dash(__name__)
# Since some components are generated dynamically, suppress callback exceptions.
app.config.suppress_callback_exceptions = True

# ------------------------- Layout Definition --------------------------------

app.layout = html.Div([
    html.H1("BayesOpt Model Predictions for Optimal Ni-Mo-Catalyst Design",
            style={'textAlign': 'center'}),
    html.H3("Andreas Burger & Paolo de Blasio", style={'textAlign': 'center'}),
    
    # ----------------- Section 1: Single Variable Analysis -----------------
    html.Div([
        html.H4("1. Single Variable Analysis", style={'textAlign': 'left', 'marginBottom': '20px'}),
        html.Div(
            children=[
                # Left column: Controls (dropdown + sliders)
                html.Div(
                    children=[
                        html.P("Select a variable to vary:", style={'textAlign': 'left'}),
                        dcc.Dropdown(
                            id='sweep-var',
                            options=[{'label': human_names[var], 'value': var} for var in variable_order],
                            value=variable_order[0],
                            style={'width': '100%'}
                        ),
                        html.P("Fix the values of the other variables:", style={'textAlign': 'left', 'marginTop': '20px'}),
                        *[
                            html.Div([
                                html.Label(f"{human_names[var]}:"),
                                dcc.Slider(
                                    id=f'slider-{var}',
                                    min=float(bounds[i, 0] if isinstance(bounds, np.ndarray) else bounds[i][0]),
                                    max=float(bounds[i, 1] if isinstance(bounds, np.ndarray) else bounds[i][1]),
                                    value=float(base_input_default[i].item()),
                                    marks={round(v, 2): str(round(v, 2))
                                           for v in np.linspace(
                                                bounds[i, 0] if isinstance(bounds, np.ndarray) else bounds[i][0],
                                                bounds[i, 1] if isinstance(bounds, np.ndarray) else bounds[i][1],
                                                5
                                            )}
                                )
                            ], style={'padding': '10px'})
                            for i, var in enumerate(variable_order)
                        ]
                    ],
                    style={'width': '30%', 'padding': '10px', 'boxSizing': 'border-box'}
                ),
                # Right column: Line plot
                html.Div(
                    children=[dcc.Graph(id='predicted-plot')],
                    style={'width': '70%', 'padding': '10px', 'boxSizing': 'border-box'}
                )
            ],
            style={'display': 'flex', 'flexDirection': 'row',
                   'border': '1px solid #ccc', 'marginBottom': '40px', 'padding': '20px'}
        )
    ]),
    
    # -------------- Section 2: Two Variable Analysis (Heatmaps) --------------
    html.Div([
        html.H4("2. Two Variable Analysis (Heatmaps)", style={'textAlign': 'left', 'marginBottom': '20px'}),
        html.Div(
            children=[
                # Left column: Dropdowns and Sliders for fixed variables.
                html.Div(
                    children=[
                        html.Div([
                            html.Label("Select Y-axis variable:"),
                            dcc.Dropdown(
                                id='heatmap-var1',
                                options=[{'label': human_names[var], 'value': var} for var in variable_order],
                                value=variable_order[0],
                                style={'width': '100%', 'marginBottom': '10px'}
                            )
                        ]),
                        html.Div([
                            html.Label("Select X-axis variable:"),
                            dcc.Dropdown(
                                id='heatmap-var2',
                                options=[{'label': human_names[var], 'value': var} for var in variable_order],
                                value=variable_order[1],
                                style={'width': '100%'}
                            )
                        ]),
                        html.Div(id='heatmap-sliders', style={'marginTop': '20px'})
                    ],
                    style={'width': '30%', 'padding': '10px', 'boxSizing': 'border-box'}
                ),
                # Right column: Two heatmaps (stacked vertically)
                html.Div(
                    children=[
                        dcc.Graph(id='predicted-heatmap', style={'marginBottom': '20px'}),
                        dcc.Graph(id='uncertainty-heatmap')
                    ],
                    style={'width': '70%', 'padding': '10px', 'boxSizing': 'border-box'}
                )
            ],
            style={'display': 'flex', 'flexDirection': 'row',
                   'border': '1px solid #ccc', 'padding': '20px', 'marginBottom': '20px'}
        )
    ])
])

# ------------------------- Callbacks ------------------------------------------

# Callback for the single variable line plot.
@app.callback(
    Output('predicted-plot', 'figure'),
    [Input('sweep-var', 'value')] +
    [Input(f'slider-{var}', 'value') for var in variable_order]
)
def update_plot(sweep_var, *values):
    """Update the line plot based on the selected sweep variable and slider values."""
    # Construct the base input vector from the slider values.
    base_input = torch.tensor(values, dtype=torch.double)
    sweep_idx = variable_order.index(sweep_var)
    
    # Generate 100 candidate values for the sweep variable.
    min_val = bounds[sweep_idx, 0] if isinstance(bounds, np.ndarray) else bounds[sweep_idx][0]
    max_val = bounds[sweep_idx, 1] if isinstance(bounds, np.ndarray) else bounds[sweep_idx][1]
    sweep_vals = torch.linspace(min_val, max_val, 100)
    
    means, stds = [], []
    for sv in sweep_vals:
        inp = base_input.clone()
        inp[sweep_idx] = sv
        posterior = best_gp.posterior(inp.unsqueeze(0))
        means.append(posterior.mean.item())
        stds.append(posterior.variance.sqrt().item())
    
    df_plot = pd.DataFrame({
        human_names[sweep_var]: sweep_vals.numpy(),
        f"Predicted {human_names[ycol]}": means,
        "Uncertainty": stds
    })
    fig = px.line(
        df_plot,
        x=human_names[sweep_var],
        y=f"Predicted {human_names[ycol]}",
        error_y="Uncertainty",
        title=f"Predicted {human_names[ycol]} vs {human_names[sweep_var]}"
    )
    fig.update_layout(
        xaxis_title=human_names[sweep_var],
        yaxis_title=f"Predicted {human_names[ycol]}",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

# Callback to dynamically create sliders for the heatmap section for variables not chosen in the dropdowns.
@app.callback(
    Output('heatmap-sliders', 'children'),
    [Input('heatmap-var1', 'value'),
     Input('heatmap-var2', 'value')]
)
def update_heatmap_sliders(var1, var2):
    sliders = []
    # Create a slider for each variable not chosen for the heatmap axes.
    for i, var in enumerate(variable_order):
        if var not in [var1, var2]:
            sliders.append(
                html.Div([
                    html.Label(f"{human_names[var]}:"),
                    dcc.Slider(
                        id={"type": "heatmap-slider", "index": var},
                        min=float(bounds[i, 0] if isinstance(bounds, np.ndarray) else bounds[i][0]),
                        max=float(bounds[i, 1] if isinstance(bounds, np.ndarray) else bounds[i][1]),
                        value=float(base_input_default[i].item()),
                        marks={round(v, 2): str(round(v, 2))
                               for v in np.linspace(
                                    bounds[i, 0] if isinstance(bounds, np.ndarray) else bounds[i][0],
                                    bounds[i, 1] if isinstance(bounds, np.ndarray) else bounds[i][1],
                                    5
                               )}
                    )
                ], style={'padding': '10px'})
            )
    return sliders

# Callback to update the two heatmaps based on dropdown and slider values.
@app.callback(
    [Output('predicted-heatmap', 'figure'),
     Output('uncertainty-heatmap', 'figure')],
    [Input('heatmap-var1', 'value'),
     Input('heatmap-var2', 'value'),
     Input({'type': 'heatmap-slider', 'index': ALL}, 'value')]
)
def update_heatmaps(var1, var2, slider_values):
    """
    Update the two heatmaps using:
      - The two variables selected in the dropdowns.
      - The fixed values (from the sliders) for the remaining variables.
    Returns two figures:
      - Predicted stability slope heatmap.
      - Uncertainty (±1σ) heatmap.
    """
    if var1 == var2:
        return go.Figure(), go.Figure()  # If the axes are the same, return empty figures.
    
    # Create base input from defaults and update with fixed slider values.
    base_input = base_input_default.clone()
    fixed_vars = [var for var in variable_order if var not in [var1, var2]]
    for var, val in zip(fixed_vars, slider_values):
        idx = variable_order.index(var)
        base_input[idx] = val

    # Create grid for the two selected variables.
    npoints = 50
    idx_y = variable_order.index(var1)
    idx_x = variable_order.index(var2)
    y_vals = torch.linspace(
        bounds[idx_y, 0] if isinstance(bounds, np.ndarray) else bounds[idx_y][0],
        bounds[idx_y, 1] if isinstance(bounds, np.ndarray) else bounds[idx_y][1],
        npoints
    )
    x_vals = torch.linspace(
        bounds[idx_x, 0] if isinstance(bounds, np.ndarray) else bounds[idx_x][0],
        bounds[idx_x, 1] if isinstance(bounds, np.ndarray) else bounds[idx_x][1],
        npoints
    )
    
    means = np.zeros((npoints, npoints))
    uncertainties = np.zeros((npoints, npoints))
    # Compute the predictions over the grid.
    for i, y_val in enumerate(y_vals):
        for j, x_val in enumerate(x_vals):
            inp = base_input.clone()
            inp[idx_y] = y_val
            inp[idx_x] = x_val
            posterior = best_gp.posterior(inp.unsqueeze(0))
            means[i, j] = posterior.mean.item()
            uncertainties[i, j] = posterior.variance.sqrt().item()
    
    # Build the predicted stability slope heatmap.
    mean_fig = go.Figure(data=go.Heatmap(
        z=means,
        x=x_vals.numpy(),
        y=y_vals.numpy(),
        colorscale='Viridis'
    ))
    mean_fig.update_layout(
        title=f'Predicted {human_names[ycol]}',
        xaxis_title=human_names[var2],
        yaxis_title=human_names[var1],
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # Build the uncertainty heatmap.
    uncert_fig = go.Figure(data=go.Heatmap(
        z=uncertainties,
        x=x_vals.numpy(),
        y=y_vals.numpy(),
        colorscale='Viridis'
    ))
    uncert_fig.update_layout(
        title='Uncertainty (±1σ)',
        xaxis_title=human_names[var2],
        yaxis_title=human_names[var1],
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    return mean_fig, uncert_fig

if __name__ == '__main__':
    app.run_server(debug=True)
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Generate sample 3D data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = []

for t in range(20):  # Time steps for the slider
    Z.append(np.sin(np.sqrt(X**2 + Y**2) + t/5) * np.exp(-0.1 * (X**2 + Y**2)))

# Create sample time series data
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100
time_series_df = pd.DataFrame({'Date': dates, 'Value': values})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Interactive Dashboard", style={'textAlign': 'center'}),
    
    # First row with heatmap and slider
    html.Div([
        html.H2("3D Data Visualization", style={'textAlign': 'center'}),
        dcc.Graph(id='heatmap'),
        html.P("Time Step:"),
        dcc.Slider(
            id='time-slider',
            min=0,
            max=19,
            value=0,
            marks={i: f't={i}' for i in range(0, 20, 2)},
            step=1
        )
    ], style={'width': '100%', 'padding': '20px'}),
    
    # Second row with additional visualizations
    html.Div([
        # Left column
        html.Div([
            html.H2("Time Series Plot", style={'textAlign': 'center'}),
            dcc.Graph(id='time-series')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Right column
        html.Div([
            html.H2("3D Surface Plot", style={'textAlign': 'center'}),
            dcc.Graph(id='surface-plot')
        ], style={'width': '48%', 'display': 'inline-block'})
    ])
])

# Callback for updating the heatmap
@app.callback(
    Output('heatmap', 'figure'),
    Input('time-slider', 'value')
)
def update_heatmap(time_step):
    fig = go.Figure(data=go.Heatmap(
        z=Z[time_step],
        x=x,
        y=y,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=f'Heatmap at t={time_step}',
        xaxis_title='X',
        yaxis_title='Y'
    )
    
    return fig

# Callback for the time series plot
@app.callback(
    Output('time-series', 'figure'),
    Input('time-slider', 'value')
)
def update_time_series(time_step):
    fig = px.line(time_series_df, x='Date', y='Value')
    fig.add_vline(x=time_series_df['Date'][time_step * 20], line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='Time Series with Current Time Step',
        xaxis_title='Date',
        yaxis_title='Value'
    )
    
    return fig

# Callback for the 3D surface plot
@app.callback(
    Output('surface-plot', 'figure'),
    Input('time-slider', 'value')
)
def update_surface(time_step):
    fig = go.Figure(data=[go.Surface(z=Z[time_step], x=x, y=y)])
    
    fig.update_layout(
        title=f'3D Surface at t={time_step}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
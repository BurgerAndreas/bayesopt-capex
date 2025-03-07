# Plot descriptions

### dashboard_beliefs.py

To analyze a single variable, we display predicted stability slope as a function of a selected variable. All parameters except the variable of interest are fixed at user-defined values, while sampling the selected parameter at 100 evenly distributed points. We compute the Gaussian Process posterior of the Bayesian optimization model at each point to obtain mean predictions and variance, rendering them as a continuous curve with ±1σ uncertainty bands.

To visialize parameter interactions we plot two variables in a heatmap. We select two parameters for variation while holding all remaining variables constant. We construct a 50×50 grid spanning the complete parameter space of the selected variables, calculating Gaussian Process predictions at each grid point. We render this data as two complementary heatmaps: one displaying predicted stability slope values and another illustrating prediction uncertainty (±1σ), enabling researchers to assess optimal catalyst compositions and prediction confidence.


### plot_bayesopt_beliefs.py
We analyze the Bayesian optimization process using various visualization techniques to reveal the model's posterior beliefs about parameter-stability relationships. We use a trained Gaussian Process (GP) model to generate predictions across the parameter space, incorporating data from multiple experiments with normalized parameters.

We visualize individual parameter relationships through one-dimensional response curves with uncertainty estimates. For each parameter, we generate predictions by varying the target parameter while fixing others at optimal values. We produce plots showing predicted mean stability slope with error bands representing one standard deviation, revealing each parameter's independent influence.

We employ heatmaps to capture parameter pair interactions on stability slope. We sample points across a grid defined by two parameters while maintaining others at optimal values. The resulting visualizations present predicted mean stability slope using a color gradient, revealing favorable stability regions.

We explicitly visualize model uncertainty through companion heatmaps displayed alongside mean prediction plots. These uncertainty heatmaps reveal regions where the model is most and least confident, with higher values indicating areas needing additional experimental data. This dual approach provides both predicted performance and confidence levels across the parameter landscape.

We combine performance and uncertainty information through scatter plots where colors represent predicted stability values and point sizes correspond to prediction uncertainty. This integrated visualization allows simultaneous evaluation of performance-confidence trade-offs. Larger points indicate higher uncertainty areas, guiding experimental design toward regions needing additional sampling.


### plot_data.py
Next we identify promising experimental conditions. We select the top 20% of data points, in terms of stability slope, from our experiments for analysis. We perform standardization using z-score normalization, with liquid variables standardized together using their common statistics.
We identify natural groupings of optimal regions by applying HDBSCAN to the standardized, high-performing points. To characterize each cluster, we calculate each cluster's centroid in the original parameter space and the associated standard deviation.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters by grouping points that exist in dense regions of the feature space. Unlike centroid-based methods like k-means, DBSCAN can be used for irregular cluster shapes and unknown number of natural groupings.
DBSCAN defines clusters as continuous regions of high point density separated by regions of low density, automatically identifying noise points that don't belong to any cluster. 
HDBSCAN extends DBSCAN by constructing a hierarchy of potential clusters at varying density levels, then extracting the most stable clusters across this hierarchy, allowing for clusters of varying densities.
"""

### plot_model_learning_progress.py
We evaluate model prediction error through a scatter plot correlating experiment number with absolute prediction error. We generate this visualization by retrieving model predictions at each optimization stage, reconstructing the Gaussian Process model's state after each iteration. We create an interactive scatter visualization with points colored by stability slope values. The horizontal axis represents chronological experiment ordering while the vertical axis quantifies prediction error magnitude, with declining trends indicating improved learning.

We calculate relative prediction error as the ratio between absolute prediction error and measured stability slope, providing insight into performance normalized by outcome magnitude. We visualize this normalized error using a similar scatter approach, mapping iterations along the x-axis and relative error along the y-axis. We color points by stability slope values for visual consistency and direct comparison. This visualization shows whether prediction accuracy improves proportionally across all outcome scales.

We track model uncertainty throughout the experimental sequence to evaluate the exploration-exploitation transition. We focus on experiments beyond the initial exploratory phase (from experiment 15) when the model leverages accumulated knowledge. We extract predicted standard deviation from the posterior distribution for each experiment. The resulting scatter plot displays experiment number versus predicted standard deviation, with points colored by stability slope values. Decreasing uncertainty over time indicates progressive transition from exploration to exploitation as confidence builds in certain parameter regions.

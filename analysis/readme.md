# Plot descriptions

You are writing the experimental section of a paper. briefly describe how the plots are generated. Write it in the style of a paper. Write one paragraph per plot. Write in full sentences, do not use bullet points or lists. Be brief. Use precise language.



### dashboard_beliefs.py
We visualize our experimental results through an interactive dashboard using Dash and Plotly frameworks to explore Bayesian optimization model predictions for Ni-Mo-catalyst design. The dashboard enables detailed analysis of stability metrics across the parameter space.

We generate line plots for single-variable analysis to display predicted stability slope as a function of a selected variable. We fix all parameters except the variable of interest at user-defined values, while sampling the selected parameter at 100 evenly distributed points. We compute the Gaussian Process posterior at each point to obtain mean predictions and variance, rendering them as a continuous curve with ±1σ uncertainty bands.

We employ heatmap visualizations for parameter interactions using a two-variable analysis approach. We select two parameters for variation while holding all remaining variables constant. We construct a 50×50 grid spanning the complete parameter space of the selected variables, calculating Gaussian Process predictions at each grid point. We render this data as two complementary heatmaps: one displaying predicted stability slope values and another illustrating prediction uncertainty (±1σ), enabling researchers to assess optimal catalyst compositions and prediction confidence.


### plot_bayesopt_beliefs.py
We analyze the Bayesian optimization process using various visualization techniques to reveal the model's posterior beliefs about parameter-stability relationships. We use a trained Gaussian Process (GP) model to generate predictions across the parameter space, incorporating data from multiple experiments with normalized parameters.

We visualize individual parameter relationships through one-dimensional response curves with uncertainty estimates. For each parameter, we generate predictions by varying the target parameter while fixing others at optimal values. We produce plots showing predicted mean stability slope with error bands representing one standard deviation, revealing each parameter's independent influence.

We employ heatmaps to capture parameter pair interactions on stability slope. We systematically sample points across a grid defined by two parameters while maintaining others at optimal values. The resulting visualizations present predicted mean stability slope using a color gradient, revealing favorable stability regions.

We explicitly visualize model uncertainty through companion heatmaps displayed alongside mean prediction plots. These uncertainty heatmaps reveal regions where the model is most and least confident, with higher values indicating areas needing additional experimental data. This dual approach provides both predicted performance and confidence levels across the parameter landscape.

We combine performance and uncertainty information through scatter plots where colors represent predicted stability values and point sizes correspond to prediction uncertainty. This integrated visualization allows simultaneous evaluation of performance-confidence trade-offs. Larger points indicate higher uncertainty areas, guiding experimental design toward regions needing additional sampling.


### plot_data.py
We generate PCA visualization by applying principal component analysis to standardized experimental data. We perform standardization using z-score normalization, with liquid variables standardized together using their common statistics. The two-dimensional embedding preserves linear relationships between input variables. We color points by stability slope values to identify favorable outcome regions.

We create UMAP plots to capture both local and global non-linear structures within the standardized data. This technique provides a complementary view to PCA by preserving topological relationships. We visualize the two-dimensional UMAP embedding as a scatter plot with points colored by stability slope values, revealing clusters not apparent in linear PCA projection.

We construct t-SNE visualization of optimal experimental regions by selecting the top 20% of data points based on stability slope. We embed these high-performing conditions into two dimensions using t-SNE with perplexity of 5. This visualization highlights the local structure of promising configurations while maintaining separation between distinct parameter regions.

We perform cluster analysis of optimal regions by applying DBSCAN to the t-SNE embedding of high-performing points. We identify natural groupings with parameters eps=0.5 and min_samples=5. We calculate each cluster's centroid in the original parameter space and annotate them on the visualization, providing insights into characteristic configurations yielding favorable stability values.

We present comparative clustering visualization through side-by-side analysis using both UMAP and t-SNE techniques. We apply HDBSCAN to the standardized data to identify natural clusters, adjusting algorithm parameters for meaningful grouping. We visualize resulting clusters in both embedding spaces, enabling robust identification of distinct parameter regimes consistently producing favorable outcomes.


### plot_model_learning_progress.py
We evaluate model prediction error through a scatter plot correlating experiment number with absolute prediction error. We generate this visualization by retrieving model predictions at each optimization stage, reconstructing the Gaussian Process model's state after each iteration. We create an interactive scatter visualization with points colored by stability slope values. The horizontal axis represents chronological experiment ordering while the vertical axis quantifies prediction error magnitude, with declining trends indicating improved learning.

We calculate relative prediction error as the ratio between absolute prediction error and measured stability slope, providing insight into performance normalized by outcome magnitude. We visualize this normalized error using a similar scatter approach, mapping iterations along the x-axis and relative error along the y-axis. We color points by stability slope values for visual consistency and direct comparison. This visualization shows whether prediction accuracy improves proportionally across all outcome scales.

We track model uncertainty throughout the experimental sequence to evaluate the exploration-exploitation transition. We focus on experiments beyond the initial exploratory phase (from experiment 15) when the model leverages accumulated knowledge. We extract predicted standard deviation from the posterior distribution for each experiment. The resulting scatter plot displays experiment number versus predicted standard deviation, with points colored by stability slope values. Decreasing uncertainty over time indicates progressive transition from exploration to exploitation as confidence builds in certain parameter regions.

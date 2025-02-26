import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import os

# import torch
# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_mll
# from botorch.optim import optimize_acqf
# from botorch.sampling import SobolQMCNormalSampler
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.models.transforms import Normalize, Standardize
# from botorch.utils.sampling import draw_sobol_samples

import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from scipy.signal import argrelextrema
from kneed import KneeLocator
import hdbscan

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# /home/andreasburger/miniforge3/envs/boc/lib/python3.10/site-packages/sklearn/utils/deprecation.py:151: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8
# /home/andreasburger/miniforge3/envs/boc/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.

from plotting_helpers import *




##################################################################
# Load data
##################################################################
current_dir = os.path.dirname(os.path.abspath(__file__))
plotfolder = os.path.join(current_dir, "plots")

# load data
data = pd.read_csv(os.path.join(current_dir, "data4.csv"), delimiter=";")

# rename the columns
data = data.rename(columns=variable_names)

ycol = "stability_slope"
xcols = [_c for _c in data.columns if _c != ycol]

# reorder the columns based on variable_order
data = data[variable_order + [ycol, "experiment"]]

##################################################################
# Just the data: PCA
##################################################################
print("")
print(f"-"*80)
# Standardization (z-score) zero mean, unit variance
# computes z = (x - mean) / std
data_standardized = StandardScaler().fit_transform(data[variable_order])

# since liquid1 and liquid2 are on the same scale, we should standardize them together
# compute the mean and std of liquid1 and liquid2
liquids_mean = np.array([data["liquid1"].mean(), data["liquid2"].mean()]).mean()
liquids_std = np.array([data["liquid1"].std(), data["liquid2"].std()]).mean()
# standardize liquid1 and liquid2 by their common mean and std
data_standardized[:, variable_order.index("liquid1")] = (data["liquid1"] - liquids_mean) / liquids_std
data_standardized[:, variable_order.index("liquid2")] = (data["liquid2"] - liquids_mean) / liquids_std

# for each column print the mean and std
# should all be close to 0 and 1, except for the liquids
print("\nStandardized data:")
for i, col in enumerate(variable_order):
    print(f"{col}: mean={data_standardized[:, i].mean()}, std={data_standardized[:, i].std()}")


# PCA preserves global structure (at least in a linear sense), distances between points are meaningful
# Only captures linear patterns (not good for complex non-linear relationships).
# PCA works by finding directions (principal components) that maximize variance in the data.

# If you apply PCA to just the six input variables (without considering the cost function), 
# you can visualize how the inputs are distributed. I.e. it tells about how the inputs were picked.
# Since most of the data is exploration, the inputs are quite uninformative.
# We could plot the inputs of only the final exploitation experiments.

# Applying PCA to just the input variables,
# will not directly tell you which regions of the input space result in a good cost.
# But we can color the PCA plot by the cost function to see if there are regions that are better or worse.
# PCA does not account for cost while computing components, 
# so there might not be clusters that align perfectly with cost variations.

pca = PCA(n_components=2)
embedding = pca.fit_transform(data_standardized)

# plot the embedding
fig = px.scatter(embedding, x=0, y=1, color=data["stability_slope"], title="PCA of data")
fig.update_layout(
    xaxis_title="PCA 1",
    yaxis_title="PCA 2",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)  # Remove whitespace around plot
)
fig.write_image(f"{plotfolder}/pca.png")
print("\nSaved pca.png")
# fig.show()
plt.close()

# Get PCA loadings (how much each variable contributes to each PC)
pca_loadings = pca.components_
# print the loadings of each variable for each component
print("\nPCA loadings:")
for i, loading in enumerate(pca_loadings):
    print(f"Weight for each input variable in PCA{i+1}: {loading}") 

# print out component "formula"
print("\nPCA component formulas:")
for i, loading in enumerate(pca_loadings):
    pca_formula = ""
    for j, var in enumerate(variable_order):
        pca_formula += f"{loading[j]:.2f} * {var} + "
    pca_formula = pca_formula[:-3]
    print(f"PCA component {i+1} formula:\n {pca_formula}")

# print the variance of each component
print(f"Variance of each component: {pca.explained_variance_}")
# # print the variance ratio of each component
# print(f"Variance ratio of each component: {pca.explained_variance_ratio_}")
# # print the cumulative variance ratio of each component
# print(f"Cumulative variance ratio of each component: {pca.explained_variance_ratio_.cumsum()}")


##################################################################
# Just the data: UMAP
##################################################################
# # UMAP preserves local and global structure approximately
# # allows for non-linear relationships

myupmap = umap.UMAP(n_components=2, random_state=42)
embedding = myupmap.fit_transform(data_standardized) 

# plot the embedding
fig = px.scatter(embedding, x=0, y=1, color=data["stability_slope"], title="UMAP of data")
fig.update_layout(
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)  # Remove whitespace around plot
)
fig.write_image(f"{plotfolder}/umap.png")
print("\nSaved umap.png")
# fig.show()
plt.close()


##################################################################
# Plot good regions 
##################################################################
# Strategy 1: dimensionality reduction + clustering
# - select the top 10% of data by stability slope (good region)
# - normalize the data
# - apply umap
# - plot the embedding
# - color by stability slope
# - optionally do kmeans clustering and plot the clusters
# - label the clusters with their unnormalized center


print("")
print(f"-"*80)
# Find natural threshold using elbow/knee method
sorted_slopes = np.sort(data['stability_slope'].values)
# Calculate differences between consecutive values
differences = np.diff(sorted_slopes)
# Find the knee point
knee_finder = KneeLocator(range(len(differences)), differences, 
                         curve='convex', direction='increasing')
threshold_idx = knee_finder.elbow
threshold = sorted_slopes[threshold_idx]

print(f"Natural threshold found at stability slope = {threshold:.3f}")
print(f"This selects {(data['stability_slope'] <= threshold).sum()} points " 
      f"({(data['stability_slope'] <= threshold).mean()*100:.1f}% of data)")

# instead compute threshhold at best 20% of data
# lower stability slope is better
threshold = sorted_slopes[int(len(sorted_slopes)*0.2)]
print(f"Threshold at best 20% of data: {threshold:.3f}")
print(f"This selects {(data['stability_slope'] <= threshold).sum()} points " 
      f"({(data['stability_slope'] <= threshold).mean()*100:.1f}% of data)")

# Select good regions based on threshold
# lower stability slope is better
good_data = data[data['stability_slope'] <= threshold].copy()
good_data_std = data_standardized[data['stability_slope'] <= threshold]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embedding = tsne.fit_transform(good_data_std)

# Plot the embedding colored by stability slope
fig = px.scatter(
    x=embedding[:, 0], 
    y=embedding[:, 1],
    color=good_data['stability_slope'],
    title="t-SNE of Good Regions",
    labels={'color': 'Stability Slope'},
    color_continuous_scale='viridis_r'
)
fig.update_layout(
    xaxis_title="t-SNE 1",
    yaxis_title="t-SNE 2",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)
)
fig.write_image(f"{plotfolder}/tsne_good_regions.png")
# fig.show()
plt.close()

# Apply DBSCAN clustering
print("\nApplying DBSCAN clustering on t-SNE embedding...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(embedding)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"Found {n_clusters} clusters")

# Plot with cluster labels
fig = px.scatter(
    x=embedding[:, 0],
    y=embedding[:, 1],
    color=clusters,
    title="t-SNE Clusters of Good Regions",
    labels={'color': 'Cluster'},
)

# Add cluster centers and statistics
for cluster_id in set(clusters):
    if cluster_id == -1:  # Skip noise points
        continue
    
    mask = clusters == cluster_id
    cluster_points = good_data[mask]
    
    # Calculate cluster center in original space
    center = cluster_points[variable_order].mean()
    avg_slope = cluster_points['stability_slope'].mean()
    
    # Add annotation with cluster info
    center_text = f"Cluster {cluster_id}:<br>"
    center_text += f"Avg slope: {avg_slope:.3f}<br>"
    for var, val in center.items():
        center_text += f"{var}: {val:.2f}<br>"
    
    # Find centroid in embedding space
    centroid = embedding[mask].mean(axis=0)
    
    fig.add_annotation(
        x=centroid[0],
        y=centroid[1],
        text=center_text,
        showarrow=True,
        arrowhead=1
    )

fig.update_layout(
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)
)
fig.write_image(f"{plotfolder}/tsne_clusters.png")
print("Saved tsne_clusters.png")
# fig.show()
plt.close()

##################################################################
# Plot good regions (better)
##################################################################
# Strategy 2: clustering + dimensionality reduction
# - Find natural threshold using elbow/knee method to select good regions
# - Normalize the data (keeping liquid1 and liquid2 on the same scale)
# - Apply HDBSCAN clustering on normalized data to find natural clusters
# - Apply UMAP to visualize the clusters in 2D
# - Print cluster statistics and centers in original space
# - Create interactive plot with:
#     - Points colored by cluster
#     - Hover text showing point details
#     - Annotations showing cluster centers and stats
#     - Side-by-side comparison with t-SNE visualization

print("")
print(f"-"*80)
# Use same threshold and standardized data from before
good_data = data[data['stability_slope'] <= threshold].copy()
good_data_std = data_standardized[data['stability_slope'] <= threshold]

# Apply HDBSCAN clustering on standardized data
print("\nApplying HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
clusters = clusterer.fit_predict(good_data_std)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"Found {n_clusters} natural clusters")
if n_clusters == 0:
    # try again with looser requirements
    print("Trying again")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3)
    clusters = clusterer.fit_predict(good_data_std)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Found {n_clusters} natural clusters")

# Print detailed cluster statistics
print("Detailed cluster statistics:")
for cluster_id in sorted(set(clusters)):
    if cluster_id == -1:
        continue
        
    mask = clusters == cluster_id
    cluster_points = good_data[mask]
    
    print(f"\nCluster {cluster_id}:")
    print(f"Number of points: {mask.sum()}")
    print(f"Average stability slope: {cluster_points['stability_slope'].mean():.3f} "
          f"± {cluster_points['stability_slope'].std():.3f}")
    print("Center point (mean of all variables):")
    for var in variable_order:
        print(f"  {var}: {cluster_points[var].mean():.2f} ± {cluster_points[var].std():.2f}")



# Apply UMAP to high-dim clustering for visualization
myumap = umap.UMAP(n_components=2, random_state=42)
umap_embedding = myumap.fit_transform(good_data_std)

# Create side-by-side plots using subplots
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=("UMAP Visualization", "t-SNE Visualization"))

# Add UMAP scatter plot
fig.add_trace(
    go.Scatter(
        x=umap_embedding[:, 0],
        y=umap_embedding[:, 1],
        mode='markers',
        marker=dict(
            color=clusters,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cluster")
        ),
        text=[f"Cluster: {c}<br>" + 
              f"Stability Slope: {s:.3f}<br>" + 
              "<br>".join([f"{var}: {val:.2f}" for var, val in row.items()]) 
              for c, s, row in zip(clusters, good_data['stability_slope'], 
                                 good_data[variable_order].to_dict('records'))],
        hoverinfo='text',
        name='UMAP'
    ),
    row=1, col=1
)

# Add t-SNE scatter plot (reuse embedding from before)
fig.add_trace(
    go.Scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        mode='markers',
        marker=dict(
            color=clusters,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f"Cluster: {c}<br>" + 
              f"Stability Slope: {s:.3f}<br>" + 
              "<br>".join([f"{var}: {val:.2f}" for var, val in row.items()]) 
              for c, s, row in zip(clusters, good_data['stability_slope'], 
                                 good_data[variable_order].to_dict('records'))],
        hoverinfo='text',
        name='t-SNE'
    ),
    row=1, col=2
)

# Add cluster statistics and centers as annotations
# for cluster_id in set(clusters):
#     if cluster_id == -1:  # Skip noise points
#         continue
        
#     mask = clusters == cluster_id
#     cluster_points = good_data[mask]
    
#     # Calculate cluster statistics
#     center = cluster_points[variable_order].mean()
#     avg_slope = cluster_points['stability_slope'].mean()
#     std_slope = cluster_points['stability_slope'].std()
    
#     # Create annotation text
#     center_text = f"Cluster {cluster_id}:<br>"
#     center_text += f"Points: {mask.sum()}<br>"
#     center_text += f"Avg slope: {avg_slope:.3f} ± {std_slope:.3f}<br>"
#     for var, val in center.items():
#         center_text += f"{var}: {val:.2f}<br>"
    
#     # Add annotations to both plots
#     for col, embedding_data in enumerate([umap_embedding, embedding], 1):
#         centroid = embedding_data[mask].mean(axis=0)
#         fig.add_annotation(
#             x=centroid[0],
#             y=centroid[1],
#             text=center_text,
#             showarrow=True,
#             arrowhead=1,
#             row=1, col=col
#         )

fig.update_layout(
    height=600,
    width=1200,
    showlegend=False,
    title_text="Comparison of UMAP and t-SNE Clustering Visualizations"
)

fig.write_image(f"{plotfolder}/clustering_comparison.png")
print("\nSaved clustering_comparison.png")
# fig.show()
plt.close()


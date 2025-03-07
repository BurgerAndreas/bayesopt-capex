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

var_names_short = {
    "Avg slope": "Y",
    "current_density": "I",
    "deposition_time": "t",
    "temperature": "T",
    "liquid1": "Liq1",
    "liquid2": "Liq2",
    "H2SO4": "H2SO4",
}

# fix random seed
np.random.seed(42)

##################################################################
# Load data
##################################################################
current_dir = os.path.dirname(os.path.abspath(__file__))
plotfolder = os.path.join(current_dir, "plots/dataplots")
os.makedirs(plotfolder, exist_ok=True)

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
"""
(didn't use these plots in the paper)
We generate PCA visualization by applying principal component analysis to standardized experimental data. The two-dimensional embedding preserves linear relationships between input variables. We color points by stability slope values to identify favorable outcome regions.
We create UMAP plots to capture both local and global non-linear structures within the standardized data. This technique provides a complementary view to PCA by preserving topological relationships. We visualize the two-dimensional UMAP embedding as a scatter plot with points colored by stability slope values, revealing clusters not apparent in linear PCA projection.
"""
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
embedding_pca = pca.fit_transform(data_standardized)

# plot the embedding
fig = px.scatter(embedding_pca, x=0, y=1, color=data["stability_slope"], title="PCA of data")
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


# -----------------------------
# Just the data: UMAP
# # UMAP preserves local and global structure approximately
# # allows for non-linear relationships

myupmap = umap.UMAP(n_components=2, random_state=42)
embedding_umap = myupmap.fit_transform(data_standardized) 

# plot the embedding
fig = px.scatter(embedding_umap, x=0, y=1, color=data["stability_slope"], title="UMAP of data")
fig.update_layout(
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)  # Remove whitespace around plot
)
figname = f"{plotfolder}/umap.png"
fig.write_image(figname)
print(f"Saved {figname}")
# fig.show()
plt.close()

# -----------------------------
# Cluster the embeddings
for embedding, method in zip([embedding_pca, embedding_umap], ["PCA", "UMAP"]):
    # Apply DBSCAN clustering with iterative parameter relaxation
    print(f"\nApplying DBSCAN clustering on {method} embedding...")
    # Start with strict parameters
    eps_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    min_samples_values = [5, 4, 3, 2]

    # Iterate through parameters until clusters are found
    clusters = None
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(embedding)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"DBSCAN with eps={eps}, min_samples={min_samples}: Found {n_clusters} clusters")
            
            # If we found clusters, break out of the loop
            if n_clusters > 0:
                print(f"Using parameters: eps={eps}, min_samples={min_samples}")
                break
        if n_clusters > 0:
            break

    # If no clusters were found with any parameters, try HDBSCAN as a fallback
    if n_clusters == 0:
        print("No clusters found with DBSCAN, trying HDBSCAN...")
        # Try HDBSCAN with progressively relaxed parameters
        hdbscan_params = [
            {"min_cluster_size": 5, "min_samples": 3},
            {"min_cluster_size": 4, "min_samples": 2},
            {"min_cluster_size": 3, "min_samples": 2},
            {"min_cluster_size": 2, "min_samples": 1}
        ]
        
        for params in hdbscan_params:
            print(f"Trying HDBSCAN with parameters: {params}")
            hdb = hdbscan.HDBSCAN(**params)
            clusters = hdb.fit_predict(embedding)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"HDBSCAN found {n_clusters} clusters")
            
            if n_clusters > 0:
                print(f"Using HDBSCAN parameters: {params}")
                break
                
        # If still no clusters found, use most relaxed parameters
        if n_clusters == 0:
            print("No clusters found with any parameters, using most relaxed settings")
            hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5)
            clusters = hdb.fit_predict(embedding)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"HDBSCAN with relaxed parameters found {n_clusters} clusters")

    for with_stats in [True, False]:
        # Plot
        fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=clusters,
            title=f"Clusters of {method} embeddings of Good Regions",
            labels={'color': 'Cluster'},
        )

        # Add cluster centers and statistics
        if with_stats:
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                mask = clusters == cluster_id
                cluster_points = data[mask]
                
                # Calculate cluster center in original space
                center = cluster_points[variable_order].mean()
                avg_slope = cluster_points['stability_slope'].mean()
                
                # Add annotation with cluster info
                # center_text = f"Cluster {cluster_id}:<br>"
                center_text = f""
                center_text += f"Y: {int(avg_slope)}<br>"
                for var, val in center.items():
                    center_text += f"{var_names_short[var]}: {val:.2f} +- {cluster_points[var].std():.2f}<br>"
                
                # Find centroid in embedding space
                centroid = embedding[mask].mean(axis=0)
                
                fig.add_annotation(
                    x=centroid[0],
                    y=centroid[1],
                    text=center_text,
                    showarrow=True,
                    arrowhead=1,
                    font=dict(size=10)  # Reduce text size
                )

        fig.update_layout(
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        figname = f"{plotfolder}/{method}_clusters{'_stats' if with_stats else ''}.png"
        fig.write_image(figname)
        print(f"Saved {figname}")
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

"""
Next we identify promising experimental conditions. We select the top 20% of data points, in terms of stability slope, from our experiments for analysis. We perform standardization using z-score normalization, with liquid variables standardized together using their common statistics.
We generate PCA visualization by applying principal component analysis to standardized experimental data. The two-dimensional embedding preserves linear relationships between input variables. We color points by stability slope values to identify favorable outcome regions.
We create UMAP plots to capture both local and global non-linear structures within the standardized data. This technique provides a complementary view to PCA by preserving topological relationships. We visualize the two-dimensional UMAP embedding as a scatter plot with points colored by stability slope values, revealing clusters not apparent in linear PCA projection.

We identify natural groupings of optimal regions by applying DBSCAN to the standardized, high-performing points. To characterize each cluster, we calculate each cluster's centroid in the original parameter space and the associated standard deviation.
"""

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

# --------------------------
# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embedding_tsne = tsne.fit_transform(good_data_std)

# Plot the embedding colored by stability slope
fig = px.scatter(
    x=embedding_tsne[:, 0], 
    y=embedding_tsne[:, 1],
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
figname = f"{plotfolder}/goodregions_tsne.png"
fig.write_image(figname)
print(f"Saved {figname}")
# fig.show()
plt.close()

# --------------------------
# Apply UMAP
myumap = umap.UMAP(n_components=2, random_state=42)
embedding_umap = myumap.fit_transform(good_data_std)

# Plot the embedding colored by stability slope
fig = px.scatter(
    x=embedding_umap[:, 0],
    y=embedding_umap[:, 1],
    color=good_data['stability_slope'],
    title="UMAP of Good Regions",
    labels={'color': 'Stability Slope'},
    color_continuous_scale='viridis_r'
)
fig.update_layout(
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)
)
figname = f"{plotfolder}/goodregions_umap.png"
fig.write_image(figname)
print(f"Saved {figname}")
# fig.show()
plt.close()


##################################################################
# DBSCAN clustering of embeddings

for embedding, method in zip([embedding_tsne, embedding_umap], ["t-SNE", "UMAP"]):
    # Apply DBSCAN clustering with iterative parameter relaxation
    print(f"\nApplying DBSCAN clustering on {method} embedding...")
    # Start with strict parameters
    eps_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    min_samples_values = [5, 4, 3, 2]

    # Iterate through parameters until clusters are found
    clusters = None
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(embedding)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"DBSCAN with eps={eps}, min_samples={min_samples}: Found {n_clusters} clusters")
            
            # If we found clusters, break out of the loop
            if n_clusters > 0:
                print(f"Using parameters: eps={eps}, min_samples={min_samples}")
                break
        if n_clusters > 0:
            break

    # If no clusters were found with any parameters, try HDBSCAN as a fallback
    if n_clusters == 0:
        print("No clusters found with DBSCAN, trying HDBSCAN...")
        # Try HDBSCAN with progressively relaxed parameters
        hdbscan_params = [
            {"min_cluster_size": 5, "min_samples": 3},
            {"min_cluster_size": 4, "min_samples": 2},
            {"min_cluster_size": 3, "min_samples": 2},
            {"min_cluster_size": 2, "min_samples": 1}
        ]
        
        for params in hdbscan_params:
            print(f"Trying HDBSCAN with parameters: {params}")
            hdb = hdbscan.HDBSCAN(**params)
            clusters = hdb.fit_predict(embedding)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"HDBSCAN found {n_clusters} clusters")
            
            if n_clusters > 0:
                print(f"Using HDBSCAN parameters: {params}")
                break
                
        # If still no clusters found, use most relaxed parameters
        if n_clusters == 0:
            print("No clusters found with any parameters, using most relaxed settings")
            hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5)
            clusters = hdb.fit_predict(embedding)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"HDBSCAN with relaxed parameters found {n_clusters} clusters")

    for with_stats in [True, False]:
        # Plot
        fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=clusters,
            title=f"Clusters of {method} embeddings of Good Regions",
            labels={'color': 'Cluster'},
        )

        # Add cluster centers and statistics
        if with_stats:
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                mask = clusters == cluster_id
                cluster_points = good_data[mask]
                
                # Calculate cluster center in original space
                center = cluster_points[variable_order].mean()
                avg_slope = cluster_points['stability_slope'].mean()
                
                # Add annotation with cluster info
                # center_text = f"Cluster {cluster_id}:<br>"
                center_text = f""
                center_text += f"Avg slope: {avg_slope:.3f}<br>"
                for var, val in center.items():
                    center_text += f"{var_names_short[var]}: {val:.2f}<br>"
                
                # Find centroid in embedding space
                centroid = embedding[mask].mean(axis=0)
                
                fig.add_annotation(
                    x=centroid[0],
                    y=centroid[1],
                    text=center_text,
                    showarrow=True,
                    arrowhead=1,
                    font=dict(size=10)  # Reduce text size
                )

        fig.update_layout(
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        figname = f"{plotfolder}/goodregions_{method}_clusters{'_stats' if with_stats else ''}.png"
        fig.write_image(figname)
        print(f"Saved {figname}")
        # fig.show()
        plt.close()

##################################################################
# Clustering good regions (better)
##################################################################
# Strategy 2: clustering + dimensionality reduction
# - Select good regions based on threshold
# - Normalize the data (keeping liquid1 and liquid2 on the same scale)
# - Apply HDBSCAN clustering on normalized data to find natural clusters

print("")
print(f"-"*80)
# Use same threshold and standardized data from before
good_data = data[data['stability_slope'] <= threshold].copy()
good_data_std = data_standardized[data['stability_slope'] <= threshold]

# Apply HDBSCAN clustering on standardized data with progressively relaxed parameters
print("\nApplying HDBSCAN clustering with multiple parameter sets...")

"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters by grouping points that exist in dense regions of the feature space. Unlike centroid-based methods like k-means, DBSCAN defines clusters as continuous regions of high point density separated by regions of low density. The algorithm requires two parameters: epsilon (ε), which defines the neighborhood radius around each point, and minPts, which specifies the minimum number of points required within that radius to form a core point. DBSCAN automatically identifies noise points that don't belong to any cluster and can discover arbitrarily shaped clusters without requiring the number of clusters to be specified in advance. This makes it particularly valuable for datasets with irregular cluster shapes and when the number of natural groupings is unknown.
HDBSCAN extends DBSCAN by constructing a hierarchy of potential clusters at varying density levels, then extracting the most stable clusters across this hierarchy. This approach eliminates the need for a fixed epsilon parameter, allowing the algorithm to identify clusters of varying densities within the same dataset while maintaining DBSCAN's ability to discover arbitrarily shaped clusters.
"""

# Define a sequence of increasingly relaxed clustering parameters
clustering_params = [
    {"min_cluster_size": 5, "min_samples": 3},
    {"min_cluster_size": 4, "min_samples": 3},
    {"min_cluster_size": 3, "min_samples": 3},
    {"min_cluster_size": 3, "min_samples": 2},
    {"min_cluster_size": 2, "min_samples": 2}
]

# Try clustering with progressively relaxed parameters until we find clusters
clusters = None
n_clusters = 0
used_params = None

for params in clustering_params:
    print(f"\nTrying HDBSCAN with parameters: {params}")
    clusterer = hdbscan.HDBSCAN(**params)
    current_clusters = clusterer.fit_predict(good_data_std)
    current_n_clusters = len(set(current_clusters)) - (1 if -1 in current_clusters else 0)
    print(f"Found {current_n_clusters} natural clusters")
    
    if current_n_clusters > 0:
        # save the first (most stringent) set of clusters
        if clusters is None:
            clusters = current_clusters
            n_clusters = current_n_clusters
            used_params = params
    
        # Print detailed cluster statistics
        print("Cluster statistics:")
        for cluster_id in sorted(set(current_clusters)):
            if cluster_id == -1:
                continue
                
            mask = current_clusters == cluster_id
            cluster_points = good_data[mask]
            
            print(f"Cluster {cluster_id}:")
            print(f" Number of points: {mask.sum()}")
            print(f" Average stability slope: {cluster_points['stability_slope'].mean():.1f} "
                f"± {cluster_points['stability_slope'].std():.3f}")
            print(" Center point (mean of all variables):")
            for var in variable_order:
                print(f"  {var}: {cluster_points[var].mean():.3f} ± {cluster_points[var].std():.3f}")

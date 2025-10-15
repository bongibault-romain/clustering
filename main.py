from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from scipy.spatial.distance import pdist
from data import arff_to_numpy
from search import grid_search
import pandas as pd
import numpy as np
from algorithm_statistics import generate_statistics, plot_comparison_table 

# Load dataset
X, _ = arff_to_numpy('dataset/artificial/dpb.arff')

mean_distance = pdist(X).mean()

print(f"Mean pairwise distance: {mean_distance}")

# Parameter grid for KMeans
param_grid = {'n_clusters': [2,3,4,5,6,7], 'init': ['k-means++', 'random']}
# results_df = grid_search(KMeans, X, param_grid)
# generate_statistics(X, results_df)

# Parameter grid for Agglomerative
param_grid = {'n_clusters': [2,3,4,5,6,7], 'metric': ['euclidean', 'l1', 'manhattan', 'cosine'], 'linkage': ['ward', 'complete', 'average']}
# results_df = grid_search(AgglomerativeClustering, X, param_grid)
# generate_statistics(X, results_df)

# Parameter grid for DBSCAN
eps_range = mean_distance * np.linspace(0.05, 0.6, 15)

param_grid = {'eps': eps_range, 'min_samples': [2,3,4,5,6], 'metric': ['euclidean', 'l1', 'manhattan', 'cosine']}
# results_df = grid_search(DBSCAN, X, param_grid)
# generate_statistics(X, results_df)

# Parameter grid for HDBSCAN
eps_range = mean_distance * np.linspace(0.05, 0.6, 15)

param_grid = {'min_cluster_size': [2,3,4,5,6], 'cluster_selection_epsilon': eps_range, 'metric': ['euclidean', 'l1', 'manhattan', 'cosine']}
results_df = grid_search(HDBSCAN, X, param_grid)
generate_statistics(X, results_df)


fig = plot_comparison_table(results_df, "HDBSCAN")
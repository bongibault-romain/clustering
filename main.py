from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
import matplotlib.pyplot as plt
from data import arff_to_numpy
from search import grid_search
import pandas as pd
import numpy as np

# Load dataset
X, _ = arff_to_numpy('dataset/artificial/long1.arff')

# Parameter grid for KMeans
param_grid = {'n_clusters': [2,3,4,5,6,7], 'init': ['k-means++', 'random']}
clusters_km, results_df = grid_search(KMeans, X, param_grid)
pd.set_option('display.max_columns', None)
print(results_df)

best_silhouette = results_df['silhouette'].max()
best_silhouette_index = results_df['silhouette'].idxmax()
best_calinski = results_df['calinski_harabasz'].max()
best_calinski_index = results_df['calinski_harabasz'].idxmax()
best_davies = results_df['davies_bouldin'].min()
best_davies_index = results_df['davies_bouldin'].idxmin()

print(f"Best silhouette: {best_silhouette} with parameters {results_df.iloc[best_silhouette_index].to_dict()}")
print(f"Best calinski_harabasz: {best_calinski} with parameters {results_df.iloc[best_calinski_index].to_dict()}")
print(f"Best davies_bouldin: {best_davies} with parameters {results_df.iloc[best_davies_index].to_dict()}")

# plot clusters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # (lignes, colonnes)
axes[0].scatter(X[:, 0], X[:, 1], c=clusters_km[best_silhouette_index], cmap='viridis', s=10)
axes[0].set_title('Best Silhouette Clusters')
axes[1].scatter(X[:, 0], X[:, 1], c=clusters_km[best_calinski_index], cmap='viridis', s=10)
axes[1].set_title('Best Calinski-Harabasz Clusters')
axes[2].scatter(X[:, 0], X[:, 1], c=clusters_km[best_davies_index], cmap='viridis', s=10)
axes[2].set_title('Best Davies-Bouldin Clusters')
plt.show()

# Parameter grid for Agglomerative
param_grid = {'n_clusters': [2,3,4,5,6,7], 'metric': ['euclidean', 'l1', 'manhattan', 'cosine'], 'linkage': ['ward', 'complete', 'average']}
cluster_ag, results_df = grid_search(AgglomerativeClustering, X, param_grid)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(results_df.sort_values(by='silhouette', ascending=False).head(10))

best_silhouette = results_df['silhouette'].max()
best_silhouette_index = results_df['silhouette'].idxmax()
best_calinski = results_df['calinski_harabasz'].max()
best_calinski_index = results_df['calinski_harabasz'].idxmax()
best_davies = results_df['davies_bouldin'].min()
best_davies_index = results_df['davies_bouldin'].idxmin()

print(f"Best silhouette: {best_silhouette} with parameters {results_df.iloc[best_silhouette_index].to_dict()}")
print(f"Best calinski_harabasz: {best_calinski} with parameters {results_df.iloc[best_calinski_index].to_dict()}")
print(f"Best davies_bouldin: {best_davies} with parameters {results_df.iloc[best_davies_index].to_dict()}")

# plot clusters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # (lignes, colonnes)
axes[0].scatter(X[:, 0], X[:, 1], c=cluster_ag[best_silhouette_index], cmap='viridis', s=10)
axes[0].set_title('Best Silhouette Clusters')
axes[1].scatter(X[:, 0], X[:, 1], c=cluster_ag[best_calinski_index], cmap='viridis', s=10)
axes[1].set_title('Best Calinski-Harabasz Clusters')
axes[2].scatter(X[:, 0], X[:, 1], c=cluster_ag[best_davies_index], cmap='viridis', s=10)
axes[2].set_title('Best Davies-Bouldin Clusters')
plt.show()

# Parameter grid for DBSCAN
param_grid = {'eps': np.linspace(0.01, 0.05), 'min_samples': [3,4,5,6,7,8], 'metric': ['euclidean', 'l1', 'manhattan', 'cosine']}
clusters_db, results_df = grid_search(DBSCAN, X, param_grid)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(results_df.sort_values(by='silhouette', ascending=False).head(10))

best_silhouette = results_df['silhouette'].max()
best_silhouette_index = results_df['silhouette'].idxmax()
best_calinski = results_df['calinski_harabasz'].max()
best_calinski_index = results_df['calinski_harabasz'].idxmax()
best_davies = results_df['davies_bouldin'].min()
best_davies_index = results_df['davies_bouldin'].idxmin()

print(f"Best silhouette: {best_silhouette} with parameters {results_df.iloc[best_silhouette_index].to_dict()}")
print(f"Best calinski_harabasz: {best_calinski} with parameters {results_df.iloc[best_calinski_index].to_dict()}")
print(f"Best davies_bouldin: {best_davies} with parameters {results_df.iloc[best_davies_index].to_dict()}")

# plot clusters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # (lignes, colonnes)
axes[0].scatter(X[:, 0], X[:, 1], c=clusters_db[best_silhouette_index], cmap='viridis', s=10)
axes[0].set_title('Best Silhouette Clusters')
axes[1].scatter(X[:, 0], X[:, 1], c=clusters_db[best_calinski_index], cmap='viridis', s=10)
axes[1].set_title('Best Calinski-Harabasz Clusters')
axes[2].scatter(X[:, 0], X[:, 1], c=clusters_db[best_davies_index], cmap='viridis', s=10)
axes[2].set_title('Best Davies-Bouldin Clusters')
plt.show()
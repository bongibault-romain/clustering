from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from scipy.spatial.distance import pdist
from data import arff_to_numpy, datasets
from search import grid_search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm_statistics import boite_a_moustache, generate_statistics, plot_comparison_table 

datasets_list = datasets(f'dataset/artificial')


for dataset in datasets_list:
    # Load dataset
    X, ground_truth = arff_to_numpy(f'dataset/artificial/{dataset}.arff')

    mean_distance = pdist(X).mean()

    # Parameter grid for KMeans
    param_grid = {'n_clusters': [2,3,4,5,6,7], 'init': ['k-means++', 'random']}
    results_df = grid_search(KMeans, X, param_grid)
    generate_statistics(X, ground_truth, results_df, save_path=f"graphics/kmeans/{dataset}")

    # Parameter grid for Agglomerative
    param_grid = {'n_clusters': [2,3,4,5,6,7], 'metric': ['euclidean', 'l1', 'manhattan'], 'linkage': ['ward', 'complete', 'average']}
    results_df = grid_search(AgglomerativeClustering, X, param_grid)
    generate_statistics(X, ground_truth, results_df, save_path=f"graphics/agglomerative/{dataset}")

    # Parameter grid for DBSCAN
    eps_range = mean_distance * np.linspace(0.05, 0.6, 15)

    param_grid = {'eps': eps_range, 'min_samples': [2,3,4,5,6], 'metric': ['euclidean', 'l1', 'manhattan', 'cosine']}
    results_df = grid_search(DBSCAN, X, param_grid)
    generate_statistics(X, ground_truth, results_df, save_path=f"graphics/dbscan/{dataset}")

    # Parameter grid for HDBSCAN
    eps_range = mean_distance * np.linspace(0.05, 2, 20)

    param_grid = {'min_cluster_size': range(2, 7), 'cluster_selection_epsilon': eps_range, 'metric': ['euclidean', 'l1', 'manhattan']}
    results_df = grid_search(HDBSCAN, X, param_grid, n_jobs=-1)
    generate_statistics(X, ground_truth, results_df, save_path=f"graphics/hdbscan/{dataset}")


    # fig = plot_comparison_table(results_df, "HDBSCAN")
    # boite_a_moustache(results_df,"HDBSCAN")
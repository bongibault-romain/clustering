from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

def safe_silhouette_score(X, labels):
    # Nombre de clusters valides (exclut le bruit -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    n_clusters = len(unique_labels)
    
    # Vérifie qu'il y a au moins 2 clusters
    if n_clusters < 2:
        return np.nan  # score invalide
    
    return silhouette_score(X, labels)

def safe_calinski_harabasz_score(X, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return np.nan  # score invalide
    
    return calinski_harabasz_score(X, labels)

def safe_davies_bouldin_score(X, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return np.nan  # score invalide
    
    return davies_bouldin_score(X, labels)
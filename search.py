from score import safe_calinski_harabasz_score, safe_davies_bouldin_score, safe_silhouette_score
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np

def grid_search(estimator, X, param_grid, scoring = {
    'silhouette': safe_silhouette_score,
    'calinski_harabasz': safe_calinski_harabasz_score,
    'davies_bouldin': safe_davies_bouldin_score
}):
    progress = 0
    results = []
    clusters = []

    for params in ParameterGrid(param_grid):
        progress += 1
        print(f"Progress: {progress}/{len(ParameterGrid(param_grid))} with parameters {params}")
        try:
            model = estimator(**params)
            labels = model.fit_predict(X)
            clusters.append(np.array(labels))

            for score_name, score_func in scoring.items():
                score = score_func(X, labels)
                entry = next((r for r in results if all(r[k] == v for k, v in params.items())), None)
                
                if entry is None:
                    entry = {**params}
                    results.append(entry)
                
                entry[score_name] = score
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            

    return clusters, pd.DataFrame(results)

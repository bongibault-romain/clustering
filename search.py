from score import safe_calinski_harabasz_score, safe_davies_bouldin_score, safe_silhouette_score
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import os
import threading
import queue
from tqdm import tqdm

def grid_search_thread(estimator, X, param_subset, scoring, queue, pbar, lock):
    results = []

    for params in param_subset:
        try:
            model = estimator(**params)
            labels = model.fit_predict(X)

            for score_name, score_func in scoring.items():
                score = score_func(X, labels)
                entry = next((r for r in results if all(r[k] == v for k, v in params.items())), None)
                
                if entry is None:
                    entry = {**params}
                    results.append(entry)
                
                entry[score_name] = score
                entry['clusters'] = labels
        except Exception as e:
            pass
            # print(f"Error with parameters {params}: {e}")
        finally:
            with lock:
                pbar.update(1)
    
    queue.put(results)

def grid_search(estimator, X, param_grid, scoring = {
    'silhouette': safe_silhouette_score,
    'calinski_harabasz': safe_calinski_harabasz_score,
    'davies_bouldin': safe_davies_bouldin_score
}, n_jobs: int = -1):
    results = []
    n_cores = os.cpu_count() if os.cpu_count() is not None else 1

    if n_jobs == -1:
        n_jobs = n_cores
    elif n_jobs < -1:
        n_jobs = max(1, n_cores + 1 + n_jobs)
    elif n_jobs == 0:
        n_jobs = 1
    elif n_jobs > n_cores:
        n_jobs = n_cores
    
    # Split the parameter of grid into n_jobs parts
    all_params = list(ParameterGrid(param_grid))
    param_grid_split = np.array_split(all_params, n_jobs)
    threads = []
    q = queue.Queue()
    lock = threading.Lock()
    
    # Initialize shared progress bar
    with tqdm(total=len(all_params), desc="Grid search progress") as pbar:
        for param_subset in param_grid_split:
            thread = threading.Thread(target=grid_search_thread,
                                      args=(estimator, X, param_subset, scoring, q, pbar, lock))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    while not q.empty():
        results.extend(q.get())

    return pd.DataFrame(results)

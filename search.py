from score import safe_calinski_harabasz_score, safe_davies_bouldin_score, safe_silhouette_score
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import os
import threading
import queue
from tqdm import tqdm
import time

def timed(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()

        # [Datetime] temps
        t = time.localtime(end)
        hours = t.tm_hour
        minutes = t.tm_min
        seconds = t.tm_sec

        print(f"[gridsearch INFO @ {hours}:{minutes}:{seconds}] Optimization took {end - start} seconds.")

        return result
    
    return wrapper

def grid_search_thread(estimator, X, param_subset, scoring, pbar: tqdm):
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
            t = time.localtime(time.time())
            hours = t.tm_hour
            minutes = t.tm_min
            seconds = t.tm_sec

            pbar.set_description(f"[gridsearch INFO @ {hours}:{minutes}:{seconds}]" )
            pbar.update(1)
    
    return results

@timed
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

    t = time.localtime(time.time())
    hours = t.tm_hour
    minutes = t.tm_min
    seconds = t.tm_sec
    
    # Initialize shared progress bar
    with tqdm(total=len(all_params), desc=f"[gridsearch INFO @ {hours}:{minutes}:{seconds}]") as pbar:
        results.extend(grid_search_thread(estimator, X, all_params, scoring, pbar))

    return pd.DataFrame(results)

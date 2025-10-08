from scipy.io import arff
import pandas as pd
import numpy as np
import os

def datasets(path):
    return np.array([f for f in os.listdir(path) if f.endswith('.arff')])

def arff_to_dataframe(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    for col in df.select_dtypes([object]).columns:
        if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, bytes)).any():
            df[col] = df[col].apply(lambda b: b.decode('utf-8') if isinstance(b, bytes) else b)

    return df

def arff_to_numpy(file_path):
    df = arff_to_dataframe(file_path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()
    return X, y
from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff('dataset/artificial/blobs.arff')
# `data` is a numpy structured array. Convert to DataFrame:
df = pd.DataFrame(data)

# If string columns appear as bytes, decode them:
for col in df.select_dtypes([object]).columns:
    if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, bytes)).any():
        df[col] = df[col].apply(lambda b: b.decode('utf-8') if isinstance(b, bytes) else b)

print(df.shape)
print(df.dtypes)
print(df.head())
# metadata:
print(meta)  # attribute names/types, etc.
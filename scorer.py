from sklearn.metrics import silhouette_score

# Define a custom scorer based on silhouette score
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)
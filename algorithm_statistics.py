import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# plt.ion()

def generate_statistics(X: np.ndarray, ground_truth: np.ndarray, results_df: pd.DataFrame, save_path: str = None):
    print(results_df.sort_values(by='silhouette', ascending=False).head(10))

    if results_df.empty:
        print("No valid clustering results to analyze.")
        return

    best_silhouette = results_df['silhouette'].max()
    best_silhouette_index = results_df['silhouette'].idxmax()
    best_calinski = results_df['calinski_harabasz'].max()
    best_calinski_index = results_df['calinski_harabasz'].idxmax()
    best_davies = results_df['davies_bouldin'].max()
    best_davies_index = results_df['davies_bouldin'].idxmax()

    print(f"Best silhouette: {best_silhouette} with parameters {results_df.iloc[best_silhouette_index].to_dict()}")
    print(f"Best calinski_harabasz: {best_calinski} with parameters {results_df.iloc[best_calinski_index].to_dict()}")
    print(f"Best davies_bouldin: {best_davies} with parameters {results_df.iloc[best_davies_index].to_dict()}")

    # plot clusters
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # (lignes, colonnes)
    axes[0].scatter(X[:, 0], X[:, 1], c=results_df.iloc[best_silhouette_index]['clusters'], cmap='viridis', s=10)
    axes[0].set_title('Best Silhouette Clusters')
    axes[1].scatter(X[:, 0], X[:, 1], c=results_df.iloc[best_calinski_index]['clusters'], cmap='viridis', s=10)
    axes[1].set_title('Best Calinski-Harabasz Clusters')
    axes[2].scatter(X[:, 0], X[:, 1], c=results_df.iloc[best_davies_index]['clusters'], cmap='viridis', s=10)
    axes[2].set_title('Best Davies-Bouldin Clusters')

    # lexicographical order of scores
    results_df = results_df.sort_values(by=['silhouette', 'calinski_harabasz', 'davies_bouldin'], ascending=[False, False, True])

    axes[3].scatter(X[:, 0], X[:, 1], c=results_df.iloc[0]['clusters'], cmap='viridis', s=10)
    axes[3].set_title('Best Overall (Lexicographically) Clusters')

    print(ground_truth)
    print(results_df.iloc[0]['clusters'])

    axes[4].scatter(X[:, 0], X[:, 1], c=ground_truth, cmap='viridis', s=10)
    axes[4].set_title('Ground Truth Clusters')

    # 10 meilleures configurations
    print("Top 10 configurations (lexicographically):")
    print(results_df.head(300))

    best_overall_index = results_df.index[0]
    # print(f"Best overall (lexicographically): {results_df.iloc[best_overall_index].to_dict()}")

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_clusters.png")
    
    plt.show()


def plot_comparison_table(results_df: pd.DataFrame, algo_name: str):
    if results_df.empty:
        print("Aucun résultat à afficher.")
        return

    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    for m in metrics:
        if m not in results_df.columns:
            print(f"Score '{m}' manquant dans les résultats.")
            return

    # Identifier les colonnes de paramètres
    param_cols = [c for c in results_df.columns if c not in metrics + ['clusters']]

    # Créer un label lisible pour chaque combinaison de paramètres
    results_df['param_label'] = results_df[param_cols].apply(
        lambda row: ', '.join([f"{k}={v}" for k, v in row.items()]), axis=1
    )

    # Sélectionner les 10 ensembles les plus "intéressants"
    # On se base sur le score de silhouette comme indicateur principal
    top_results = results_df.sort_values(by='silhouette', ascending=False)

    # Positions des barres
    x = np.arange(len(top_results))

    # Création des 3 graphiques côte à côte
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Comparaison des métriques pour {algo_name}", fontsize=16, fontweight='bold')

    # Silhouette
    axes[0].bar(x, top_results['silhouette'], color='teal')
    axes[0].set_title("Silhouette Score", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_ylabel("Score")
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Davies-Bouldin
    axes[1].bar(x, top_results['davies_bouldin'], color='orange')
    axes[1].set_title("Davies-Bouldin Index", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_ylabel("Score (plus bas = mieux)")
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Calinski-Harabasz
    axes[2].bar(x, top_results['calinski_harabasz'], color='purple')
    axes[2].set_title("Calinski-Harabasz Score", fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_ylabel("Score (plus haut = mieux)")
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def boite_a_moustache(results_df: pd.DataFrame, algo_name: str):
    if results_df.empty:
        print("Aucun résultat à afficher.")
        return
    
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    for m in metrics:
        if m not in results_df.columns:
            print(f"Score '{m}' manquant dans les résultats.")
            return

    # Création de 3 axes côte à côte
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Distribution des métriques pour {algo_name}", fontsize=16, fontweight='bold')

    # Silhouette
    axes[0].boxplot(results_df['silhouette'])
    axes[0].set_title("Silhouette")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Davies-Bouldin
    axes[1].boxplot(results_df['davies_bouldin'])
    axes[1].set_title("Davies-Bouldin")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Calinski-Harabasz
    axes[2].boxplot(results_df['calinski_harabasz'])
    axes[2].set_title("Calinski-Harabasz")
    axes[2].set_ylabel("Score")
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
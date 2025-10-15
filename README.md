### Romain Bongibault
### Elsa Hindi
---

# TP1 - Clustering

## Compréhension des méthodes et hyperparamètres
1- KMeans:

Partitionne les données en k clusters, chaque point appartenant au cluster dont la moyenne (centre/centroïde) est la plus proche. Minimise la somme des carrés intra-cluster.

| Hyperparamètres étudiés | Raison |
| ---- | --- |
| n_clusters   | Un mauvais choix de k peut entraîner une sous-segmentation ou une sur-segmentation des données. | 
| init    | Une mauvaise initialisation (comme 'random') peut conduire l'algorithme à un minimum local sous-optimal.     |




2- Clustering Hiérarchique (AgglomerativeClustering) :

Construit une hiérarchie de clusters. Part d'un cluster par point, puis fusionne itérativement les paires de clusters les plus proches jusqu'à atteindre le nombre souhaité.

| Hyperparamètres étudiés | Raison |
| ---- | --- |
| n_clusters   |Le nombre cible de clusters à former en coupant l'arbre (dendrogramme) résultant. | 
| linkage   | Définit la métrique de distance entre deux clusters.       |
| metric  | La métrique de distance utilisée pour calculer la proximité entre les points (par exemple, 'euclidean' ou 'manhattan'). | 


3- DBSCAN : 

Regroupe les points fortement connectés par la densité. Identifie les régions denses et les sépare du bruit.

| Hyperparamètres étudiés | Raison |
| ---- | --- |
| eps   | Le rayon du voisinage. C'est l'hyperparamètre le plus critique : il détermine la distance maximale pour que deux points soient considérés comme voisins. Trop petit, il crée trop de bruit; trop grand, il fusionne des clusters distincts. | 
| min_samples   | Le nombre minimum de points requis pour former une région dense (point central). Il contrôle la sensibilité de l'algorithme au bruit.  | | 

4- HDBSCAN : 

Version hiérarchique de DBSCAN. Construit une hiérarchie de densité (dont la structure fondamentale repose sur l'Arbre Couvrant Minimum) et extrait les clusters les plus stables sur différentes échelles de densité.

| Hyperparamètres étudiés | Raison |
| ---- | --- |
| min_cluster_size   | Définit la plus petite taille de cluster considérée comme significative. Il a un impact direct sur le nombre total de clusters trouvés. | 
| min_samples   | L'équivalent de MinPts. Il influence l'estimation de la densité et le bruit. Une valeur élevée rend l'algorithme plus sensible au bruit.     |
| cluster_selection_method | Définit comment les clusters sont extraits de la hiérarchie condensée. 'eom' sélectionne les clusters qui existent le plus longtemps (les plus stables), essentiel pour la détection de densité variable.| 

## Limite des méthodes

KMeans : 

- Trouver K à l’avance: le choix de cette valeur est très difficile à estimer. Souvent, on ne sait pas à l’avance en combien de catégories on doit diviser un ensemble de données
- Quantité de données très grande = coût en temps de l’algorithme est très grand
- Si le cluster contient des points éloignés, la valeur moyenne dévie sérieusement
- Ne convient pas à la découverte de clusters de formes non convexes ou de clusters de tailles très différentes

Clustering Hiérarchique:
- Pas idéal pour les grands ensembles de données
- Sensible au bruit et aux valeurs aberrantes: Quelques valeurs aberrantes peuvent fausser le dendrogramme de manière significative
- Pas de réaffectation: Une fois qu'une fusion est effectuée, elle ne peut être annulée, ce qui peut conduire à un regroupement sous-optimal.

DBSCAN : 
- Densité variable dans les données
- Grandes dimensions (temps de calcul)
- Difficultés à déterminer les paramètres : Fixer la taille du voisinage et le
nombre de voisins à considérer

HDBSCAN : 

- Coût de Construction de la Hiérarchie : HDBSCAN doit calculer la distance de mutual reachability pour toutes les paires de points et construire l'Arbre Couvrant Minimum (MST). Dans le pire des cas, cela a une complexité d'environ O(n^2), où n est le nombre de points de données.

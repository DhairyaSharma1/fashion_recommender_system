import pickle
import numpy as np
import collections

# Load existing embeddings and cluster labels
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
cluster_labels = pickle.load(open('dbscan_labels.pkl', 'rb'))

# Compute and save cluster centroids
cluster_to_features = collections.defaultdict(list)
for label, feature in zip(cluster_labels, feature_list):
    if label != -1:  # Ignore outliers
        cluster_to_features[label].append(feature)

cluster_centroids = {}
for cluster_id, features in cluster_to_features.items():
    cluster_centroids[cluster_id] = np.mean(features, axis=0)

# Save cluster centroids
pickle.dump(cluster_centroids, open("cluster_centroids.pkl", "wb"))
print("Cluster centroids saved as 'cluster_centroids.pkl'")

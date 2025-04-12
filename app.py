import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import hnswlib
from sklearn.cluster import DBSCAN
import collections

# Load and modify model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Extract features
filenames = [os.path.join('fashion-dataset/images', file) for file in os.listdir('fashion-dataset/images')]
feature_list = [extract_features(file, model) for file in tqdm(filenames)]

# Save features
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

# Clustering with DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
cluster_labels = dbscan.fit_predict(feature_list)

# Save DBSCAN model and cluster labels
pickle.dump(dbscan, open('dbscan.pkl', 'wb'))
pickle.dump(cluster_labels, open('dbscan_labels.pkl', 'wb'))

# Compute and save cluster centroids
cluster_to_features = collections.defaultdict(list)
for label, feature in zip(cluster_labels, feature_list):
    if label != -1:  # Ignore outliers
        cluster_to_features[label].append(feature)

cluster_centroids = {}
for cluster_id, features in cluster_to_features.items():
    cluster_centroids[cluster_id] = np.mean(features, axis=0)

pickle.dump(cluster_centroids, open("cluster_centroids.pkl", "wb"))

# Build and save HNSWlib index
dim = len(feature_list[0])
p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=len(feature_list), ef_construction=200, M=16)
p.add_items(np.array(feature_list), np.arange(len(feature_list)))
p.save_index("hnsw_index.bin")

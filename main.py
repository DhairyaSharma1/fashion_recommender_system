from PIL import Image, UnidentifiedImageError
import streamlit as st
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import hnswlib

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("👗 Fashion Recommender System")

# Load files
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))
cluster_labels = pickle.load(open("dbscan_labels.pkl", "rb"))
cluster_centroids = pickle.load(open("cluster_centroids.pkl", "rb"))

# Load model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img, verbose=0).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def assign_to_cluster(features):
    min_dist = float('inf')
    best_cluster = -1
    for cluster_id, centroid in cluster_centroids.items():
        dist = np.linalg.norm(features - centroid)
        if dist < min_dist:
            min_dist = dist
            best_cluster = cluster_id
    return best_cluster, min_dist

def recommend(features, top_k=6, dist_threshold=0.6):
    cluster_id, min_dist = assign_to_cluster(features)

    if min_dist > dist_threshold:
        st.warning("The uploaded image doesn't match any known cluster well. Results may be less accurate.")

    # Get images from same cluster
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    cluster_embeddings = [feature_list[i] for i in cluster_indices]
    cluster_filenames = [filenames[i] for i in cluster_indices]

    # Build temporary index for fast search
    temp_index = hnswlib.Index(space='l2', dim=len(features))
    temp_index.init_index(max_elements=len(cluster_embeddings), ef_construction=100, M=16)
    temp_index.add_items(np.array(cluster_embeddings), np.arange(len(cluster_embeddings)))

    labels, distances = temp_index.knn_query(features, k=top_k)

    return [cluster_filenames[i] for i in labels[0]]

# Streamlit UI
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, etc.)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # Open uploaded image
        image_data = Image.open(uploaded_file)

        # Resize the uploaded image to 224x224 to match the recommended images' size
        image_data = image_data.resize((224, 224))  # Resize to fixed size of 224x224

        # Display the uploaded image and recommended results in a compact layout
        st.markdown("<h3>Uploaded Image</h3>", unsafe_allow_html=True)  # Left-aligned heading

        # Layout: Display the uploaded image and recommendations in the same row
        col1, col2 = st.columns([2, 6])  # Adjust column width ratios

        with col1:
            st.image(image_data, caption="Uploaded Image", use_container_width=True)

        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            features = feature_extraction(file_path, model)
            if features is not None:
                results = recommend(features)

                st.subheader("🧠 Recommended Results:")
                # Create a layout with multiple columns for recommended images
                cols = st.columns(5)

                # Display each recommended image inside the columns
                for i, col in enumerate(cols):
                    with col:
                        if i < len(results):
                            st.image(results[i], caption=f"Recommendation {i+1}", use_container_width=True)

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please try re-saving it as PNG or JPG.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

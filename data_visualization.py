import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
import ast
from google.cloud import storage
import os
import tempfile
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


# ---------------------------
# Configuration
# ---------------------------

# Path to your CSV file
CSV_PATH = 'embeddings_data.csv'  # Ensure this path is correct

# Path to your Google Cloud service account key
SERVICE_ACCOUNT_KEY = 'gcs-service-account-key.json'  # Ensure this path is correct

# Google Cloud Storage bucket name
BUCKET_NAME = 'openvideos'

# ---------------------------
# Data Loading and Preparation
# ---------------------------

@st.cache_data
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

    # Function to parse embeddings
    def parse_embeddings(embedding_str):
        try:
            return ast.literal_eval(embedding_str)
        except Exception as e:
            st.error(f"Error parsing embeddings: {e}")
            return []

    # Apply parsing
    df['video_embeddings'] = df['video_embeddings'].apply(parse_embeddings)

    # Expand video embeddings if multiple segments per video
    df_expanded = df.explode('video_embeddings').reset_index(drop=True)

    # Extract the embedding vector from video_embeddings
    df_expanded['video_embedding'] = df_expanded['video_embeddings'].apply(
        lambda x: x['embedding'] if isinstance(x, dict) and 'embedding' in x else []
    )

    # Optionally, keep start and end times
    df_expanded['start_offset_sec'] = df_expanded['video_embeddings'].apply(
        lambda x: x.get('start_offset_sec') if isinstance(x, dict) else None
    )
    df_expanded['end_offset_sec'] = df_expanded['video_embeddings'].apply(
        lambda x: x.get('end_offset_sec') if isinstance(x, dict) else None
    )

    # Drop the intermediate column
    df_expanded = df_expanded.drop(columns=['video_embeddings'])

    # Prepare video DataFrame
    df_video = df_expanded[['video_embedding', 'video_path', 'start_offset_sec', 'end_offset_sec']].copy()
    df_video = df_video.dropna(subset=['video_embedding'])
    df_video['type'] = 'video'

    # Aggregate video embeddings per video_path
    df_video_grouped = df_video.groupby('video_path')['video_embedding'].apply(list).reset_index()

    # Compute centroid (average) for each video's embeddings and normalize
    def aggregate_and_normalize(embeddings):
        if not embeddings:
            return np.array([])
        embeddings_array = np.vstack(embeddings)  # Shape: (num_segments, embedding_dim)
        embeddings_normalized = normalize(embeddings_array)  # L2 normalize each segment
        centroid = np.mean(embeddings_normalized, axis=0)  # Compute centroid
        return centroid

    df_video_grouped['video_embedding_normalized'] = df_video_grouped['video_embedding'].apply(aggregate_and_normalize)

    # Remove any rows with empty embeddings
    df_video_grouped = df_video_grouped[
        df_video_grouped['video_embedding_normalized'].apply(lambda x: x.size > 0)
    ].reset_index(drop=True)

    # Convert embeddings to numpy arrays (ensure consistent dimensionality)
    df_video_grouped['video_embedding_normalized'] = df_video_grouped['video_embedding_normalized'].apply(lambda x: np.array(x))

    # Verify that all embeddings have the same dimension
    video_dim = df_video_grouped['video_embedding_normalized'].apply(lambda x: x.shape[0]).unique()

    if len(video_dim) > 1:
        st.error("Inconsistent embedding dimensions found in the data.")
        st.stop()

    return df_video_grouped

@st.cache_data
def perform_tsne(embeddings, n_components=2, random_state=42, perplexity=30, n_iter=1000):
    """
    Performs t-SNE dimensionality reduction on the embeddings.

    Parameters:
    - embeddings (np.ndarray): Array of shape (n_samples, n_features).
    - n_components (int): Number of dimensions for t-SNE.
    - random_state (int): Seed for reproducibility.
    - perplexity (float): Perplexity parameter for t-SNE.
    - n_iter (int): Number of iterations for optimization.

    Returns:
    - np.ndarray: Array of shape (n_samples, n_components) with t-SNE results.
    """
    print("Initializing t-SNE...")
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, n_iter=n_iter, verbose=1)
    print("Performing t-SNE dimensionality reduction. This may take a while...")
    tsne_results = tsne.fit_transform(embeddings)
    print("t-SNE completed.")
    return tsne_results

@st.cache_resource
def init_storage(service_account_key, bucket_name):
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        return bucket
    except Exception as e:
        st.error(f"Error initializing Google Cloud Storage: {e}")
        return None

def fetch_video(bucket, video_path):
    try:
        blob = bucket.blob(video_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            blob.download_to_filename(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error fetching video from GCS: {e}")
        return None

# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.set_page_config(page_title="Embeddings Visualization", layout="wide")
    st.title(" Video Embeddings Visualization")

    # Display a banner or description
    st.markdown("""
    This application visualizes video embeddings in a 2D space using t-SNE.
    - **Video Embeddings** are represented as blue points labeled as *video*.

    Select a point from the dropdown below to view the associated video.
    """)

    # Load data
    st.header("Loading and Preparing Data")
    with st.spinner('Loading and processing data...'):
        df_video = load_data(CSV_PATH)

    if df_video.empty:
        st.warning("No valid video embeddings found in the CSV file.")
        return

    st.success("Data loaded and processed successfully!")

    # Combine video embeddings for t-SNE
    video_embeddings = np.vstack(df_video['video_embedding_normalized'].tolist())

    # Validate video_embeddings
    st.write(f"Shape of video_embeddings: {video_embeddings.shape}")
    if np.isnan(video_embeddings).any():
        st.error("Embeddings contain NaN values. Please check your data.")
        return
    if np.isinf(video_embeddings).any():
        st.error("Embeddings contain Inf values. Please check your data.")
        return

    # Perform t-SNE
    st.header("Performing t-SNE Dimensionality Reduction")
    with st.spinner('Reducing dimensionality with t-SNE...'):
        try:
            embedding_2d = perform_tsne(video_embeddings, perplexity=30, n_iter=1000)
        except Exception as e:
            st.error(f"Error during t-SNE dimensionality reduction: {e}")
            return

    # Assign to DataFrame
    df_video['tsne_x'] = embedding_2d[:, 0]
    df_video['tsne_y'] = embedding_2d[:, 1]

    # Prepare DataFrame for plotting
    df_plot = df_video[['tsne_x', 'tsne_y', 'video_path']].copy()
    df_plot['type'] = 'video'

    st.write(f"Shape of df_plot: {df_plot.shape}")
    st.write("Sample data from df_plot:")
    st.dataframe(df_plot.head())

    # Initialize Google Cloud Storage
    st.header("Initializing Google Cloud Storage")
    bucket = init_storage(SERVICE_ACCOUNT_KEY, BUCKET_NAME)
    if bucket is None:
        st.error("Failed to initialize Google Cloud Storage. Please check your credentials.")
        return

    st.success("Google Cloud Storage initialized successfully!")

    # Plot using Plotly
    st.header("t-SNE Projection of Video Embeddings")
    try:
        fig = px.scatter(
            df_plot,
            x='tsne_x',
            y='tsne_y',
            color='type',
            hover_data=['video_path'],
            title='t-SNE Projection of Video Embeddings',
            width=1200,
            height=600,
            template='plotly_dark',
            symbol='type',
            symbol_map={'video': 'circle'}
        )
    except Exception as e:
        st.error(f"Error creating scatter plot: {e}")
        return

    # Update marker styles for better visibility
    fig.update_traces(marker=dict(size=8, opacity=0.7),
                      selector=dict(mode='markers'))

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Interactive selection via dropdown
    st.header("Select a Video to View")

    # Create a list of tuples with (index, video_path)
    options = []
    for idx, row in df_plot.iterrows():
        # Create a concise label for each point
        label_video = row['video_path'].split('/')[-1]  # Display only the filename
        label = f"Video | {label_video}"
        options.append((idx, label))

    # Create a dictionary for easy lookup
    options_dict = {label: idx for idx, label in options}

    # Dropdown selection
    selected_label = st.selectbox("Select a video:", [label for label, idx in options])

    if selected_label:
        selected_index = options_dict[selected_label]
        row = df_plot.iloc[selected_index]

        st.subheader(f"Details of Selected Video (Index: {selected_index})")
        st.write(f"**Video Path:** {row['video_path']}")

        # Fetch and display video
        video_local_path = fetch_video(bucket, row['video_path'])
        if video_local_path:
            st.video(video_local_path)
        else:
            st.error("Failed to fetch the video.")

    # Optional: Display the total number of points
    st.sidebar.markdown(f"**Total Videos:** {len(df_plot)}")

if __name__ == "__main__":
    main()
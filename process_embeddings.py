import pandas as pd
import numpy as np
import ast
import os
import umap
import json
from sklearn.cluster import DBSCAN

# ---------------------------
# Configuration
# ---------------------------

# Path to your CSV file containing embeddings
CSV_PATH = 'embeddings_data.csv'

# Output paths for UMAP embeddings
VIDEO_UMAP_OUTPUT = 'video_umap_embeddings_dbscan_08.csv'

# DBSCAN Configuration
EPS = 0.8  # The maximum distance between two samples for them to be considered as in the same neighborhood.
MIN_SAMPLES = 5  # The number of samples in a neighborhood for a point to be considered as a core point.

# GCS Public URL format
GCS_PUBLIC_URL_TEMPLATE = 'https://storage.googleapis.com/{bucket}/{path}'

# ---------------------------
# Data Loading and Preparation
# ---------------------------

def load_video_embeddings(csv_path):
    """
    Load and process video embeddings from the CSV file.

    Parameters:
    - csv_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame with video_path, normalized embeddings, video_url, and cluster labels.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

    # Function to parse video embeddings
    def parse_video_embeddings(embedding_str):
        try:
            return ast.literal_eval(embedding_str)
        except Exception as e:
            print(f"Error parsing video embeddings: {e}")
            return []

    # Apply parsing
    df['video_embeddings'] = df['video_embeddings'].apply(parse_video_embeddings)

    # Expand video embeddings if multiple segments per video
    df_expanded = df.explode('video_embeddings').reset_index(drop=True)

    # Extract the embedding vector from video_embeddings
    df_expanded['video_embedding'] = df_expanded['video_embeddings'].apply(
        lambda x: x['embedding'] if isinstance(x, dict) and 'embedding' in x else []
    )

    # Drop the intermediate column
    df_expanded = df_expanded.drop(columns=['video_embeddings'])

    # Prepare video DataFrame
    df_video = df_expanded[['video_embedding', 'video_path']].copy()
    df_video = df_video.dropna(subset=['video_embedding'])
    df_video['video_embedding'] = df_video['video_embedding'].apply(lambda x: np.array(x))

    # Aggregate video embeddings per video_path by computing the centroid
    df_video_grouped = df_video.groupby('video_path')['video_embedding'].apply(
        lambda embeddings: np.mean(np.vstack(embeddings), axis=0)
    ).reset_index()

    # Normalize the embeddings
    df_video_grouped['video_embedding_normalized'] = df_video_grouped['video_embedding'].apply(
        lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x
    )

    # Generate Public URLs
    bucket_name = 'openvideos'  # Your GCS bucket name
    df_video_grouped['video_url'] = df_video_grouped['video_path'].apply(
        lambda path: GCS_PUBLIC_URL_TEMPLATE.format(
            bucket=bucket_name,
            path=path.replace(f'gs://{bucket_name}/', '')
        )
    )
    
    # Optional: Verify URL format (you can remove this if not needed)
    # df_video_grouped['video_url'] = df_video_grouped['video_url'].apply(
    #     lambda url: url if url.startswith('https://storage.googleapis.com/') else ''
    # )

    # Perform DBSCAN Clustering
    if not df_video_grouped['video_embedding_normalized'].empty:
        print(f"Performing DBSCAN clustering with eps={EPS} and min_samples={MIN_SAMPLES} on video embeddings...")
        video_embeddings = np.vstack(df_video_grouped['video_embedding_normalized'].values)
        dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
        df_video_grouped['video_cluster'] = dbscan.fit_predict(video_embeddings)
        
        # Calculate number of clusters excluding noise (-1)
        unique_clusters = set(df_video_grouped['video_cluster'])
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        print(f"DBSCAN clustering completed with {n_clusters} clusters.")
    else:
        df_video_grouped['video_cluster'] = np.nan
        print("No video embeddings available for clustering.")

    return df_video_grouped[['video_path', 'video_embedding_normalized', 'video_url', 'video_cluster']]

# ---------------------------
# UMAP Dimensionality Reduction
# ---------------------------

def compute_umap(embeddings, n_components=2, random_state=42):
    """
    Compute UMAP embeddings.

    Parameters:
    - embeddings (np.ndarray): Array of shape (n_samples, n_features).
    - n_components (int): Number of dimensions for UMAP.
    - random_state (int): Seed for reproducibility.

    Returns:
    - np.ndarray: UMAP reduced embeddings.
    """
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

# ---------------------------
# Saving Embeddings with Clusters
# ---------------------------

def save_umap_embeddings(df, umap_embeddings, id_column, output_path):
    """
    Save UMAP embeddings to a CSV file along with cluster assignments and video URLs.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the identifiers, cluster labels, and video URLs.
    - umap_embeddings (np.ndarray): UMAP reduced embeddings.
    - id_column (str): Column name for the identifier.
    - output_path (str): Path to save the CSV file.
    """
    df_umap = pd.DataFrame(umap_embeddings, columns=['umap_x', 'umap_y'])
    df_umap[id_column] = df[id_column].values

    if 'video_cluster' in df.columns:
        df_umap['cluster'] = df['video_cluster']
    else:
        df_umap['cluster'] = np.nan

    if 'video_url' in df.columns:
        df_umap['video_url'] = df['video_url'].values
    else:
        df_umap['video_url'] = ''

    # Reorder columns to include video_url
    df_umap = df_umap[[id_column, 'umap_x', 'umap_y', 'cluster', 'video_url']]
    df_umap.to_csv(output_path, index=False)
    print(f"UMAP embeddings with cluster assignments and video URLs saved to {output_path}")

# ---------------------------
# Main Execution
# ---------------------------

def main():
    # Load and process video embeddings
    print("Loading video embeddings...")
    df_video = load_video_embeddings(CSV_PATH)
    if df_video.empty:
        print("No video embeddings to process.")
    else:
        print(f"Video embeddings shape: {df_video.shape}")

        # Compute UMAP for video embeddings
        print("Computing UMAP for video embeddings...")
        video_umap = compute_umap(np.vstack(df_video['video_embedding_normalized'].values))
        save_umap_embeddings(df_video, video_umap, 'video_path', VIDEO_UMAP_OUTPUT)

if __name__ == "__main__":
    main() 
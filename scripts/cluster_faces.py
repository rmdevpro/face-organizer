#!/usr/bin/env python3
"""
Cluster face embeddings into Person A and Person B using K-Means
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import faiss

def cluster_embeddings(embedding_files, output_dir, n_clusters=2):
    """
    Cluster embeddings and organize images into folders

    Args:
        embedding_files: List of pickle files containing embeddings
        output_dir: Output directory for clustered images
        n_clusters: Number of clusters (default: 2 for Person A and B)
    """
    print(f"\n{'='*80}")
    print(f"FACE CLUSTERING WITH K-MEANS")
    print(f"{'='*80}\n")

    # Load all embeddings
    all_embeddings = []
    all_image_paths = []
    total_no_face = 0
    total_images = 0

    for emb_file in embedding_files:
        print(f"Loading {emb_file}...")
        with open(emb_file, 'rb') as f:
            data = pickle.load(f)
            all_embeddings.append(data['embeddings'])
            all_image_paths.extend(data['image_paths'])
            total_no_face += data['no_face_count']
            total_images += data['total_images']

    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)

    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Detection rate: {len(embeddings)}/{total_images} ({len(embeddings)/total_images*100:.1f}%)")
    print(f"No face detected: {total_no_face}\n")

    # Normalize embeddings (L2 norm) - important for cosine similarity
    faiss.normalize_L2(embeddings)

    # Run K-Means clustering using FAISS
    print(f"Running K-Means clustering (k={n_clusters})...")

    # Create K-Means clusterer
    d = embeddings.shape[1]  # Dimension
    kmeans = faiss.Kmeans(d, n_clusters, niter=50, verbose=True, gpu=False)

    # Train
    kmeans.train(embeddings)

    # Assign each embedding to nearest cluster
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()

    # Count cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster sizes:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(labels) * 100)
        print(f"  Cluster {cluster_id}: {count} images ({percentage:.1f}%)")

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cluster_dirs = {}
    for cluster_id in unique:
        cluster_dir = output_path / f"person_{chr(65 + cluster_id)}"  # A, B, C, ...
        cluster_dir.mkdir(exist_ok=True)
        cluster_dirs[cluster_id] = cluster_dir

    # Copy images to cluster folders
    print(f"\nCopying images to cluster folders...")
    for img_path, cluster_id in tqdm(zip(all_image_paths, labels), total=len(labels)):
        src = Path(img_path)
        dst = cluster_dirs[cluster_id] / src.name

        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error copying {src}: {e}")

    print(f"\n{'='*80}")
    print(f"CLUSTERING COMPLETE")
    print(f"{'='*80}\n")

    for cluster_id in unique:
        cluster_dir = cluster_dirs[cluster_id]
        count = len(list(cluster_dir.glob('*.png'))) + \
                len(list(cluster_dir.glob('*.jpg'))) + \
                len(list(cluster_dir.glob('*.jpeg')))
        print(f"  person_{chr(65 + cluster_id)}: {count} images → {cluster_dir}")

    # Save cluster assignments for reference
    assignment_file = output_path / "cluster_assignments.pkl"
    with open(assignment_file, 'wb') as f:
        pickle.dump({
            'labels': labels,
            'image_paths': all_image_paths,
            'cluster_centers': kmeans.centroids
        }, f)

    print(f"\n  Cluster assignments saved to: {assignment_file}")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster face embeddings into Person A and B')
    parser.add_argument('--embeddings', '-e', nargs='+', required=True,
                       help='Pickle files containing embeddings')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for clustered images')
    parser.add_argument('--clusters', '-k', type=int, default=2,
                       help='Number of clusters (default: 2)')

    args = parser.parse_args()

    cluster_embeddings(args.embeddings, args.output, args.clusters)

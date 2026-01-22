#!/usr/bin/env python3
"""
FAISS vector database operations.
Builds, saves, loads, and searches FAISS indexes for image similarity.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np


class ImageVectorDB:
    """FAISS-based vector database for image similarity search."""

    def __init__(self, dim=None):
        """
        Initialize the vector database.

        Args:
            dim: Embedding dimension (if known beforehand)
        """
        self.index = None
        self.image_paths = []
        self.dim = dim

    def build_index(self, embeddings, image_paths, metric='ip'):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: numpy array [N, dim] of embeddings
            image_paths: List of image paths corresponding to embeddings
            metric: 'ip' (Inner Product) or 'l2' (Euclidean distance)

        Returns:
            FAISS index
        """
        self.image_paths = [str(p) for p in image_paths]
        self.dim = embeddings.shape[1]

        if metric == 'ip':
            # Inner Product on normalized vectors = Cosine Similarity
            self.index = faiss.IndexFlatIP(self.dim)
        elif metric == 'l2':
            # L2 distance
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'ip' or 'l2'")

        self.index.add(embeddings.astype('float32'))

        print(f"Built FAISS index with {self.index.ntotal} vectors (dim={self.dim})")
        return self.index

    def search(self, query_embedding, k=10):
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector [dim] or [1, dim]
            k: Number of results to return

        Returns:
            List of tuples: [(image_path, score), ...]
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Reshape to [1, dim] if needed
        query_vec = np.array(query_embedding, dtype='float32')
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.image_paths) and idx >= 0:
                results.append((self.image_paths[idx], float(dist)))

        return results

    def save(self, prefix):
        """
        Save index and image paths to disk.

        Args:
            prefix: File path prefix (e.g., "image_db" -> "image_db.faiss", "image_db_paths.pkl")
        """
        if self.index is None:
            raise RuntimeError("No index to save.")

        index_path = f"{prefix}.faiss"
        paths_path = f"{prefix}_paths.pkl"

        faiss.write_index(self.index, index_path)

        with open(paths_path, "wb") as f:
            pickle.dump(self.image_paths, f)

        print(f"Saved index to {index_path}")
        print(f"Saved image paths to {paths_path}")

    def load(self, prefix):
        """
        Load index and image paths from disk.

        Args:
            prefix: File path prefix used when saving
        """
        index_path = f"{prefix}.faiss"
        paths_path = f"{prefix}_paths.pkl"

        if not os.path.exists(index_path) or not os.path.exists(paths_path):
            raise FileNotFoundError(
                f"Index files not found: {index_path}, {paths_path}"
            )

        self.index = faiss.read_index(index_path)

        with open(paths_path, "rb") as f:
            self.image_paths = pickle.load(f)

        self.dim = self.index.d

        print(f"Loaded index with {self.index.ntotal} vectors (dim={self.dim})")

"""
Image similarity search using CAV and FAISS.

This package provides tools for building image vector databases and
searching for similar images using Concept Activation Vectors.
"""

from src.lib.cav import (
    compute_cav,
    compute_cav_from_images,
    compute_word_cav,
    get_image_similarity_scores,
    get_word_similarity_score,
    normalize_cav,
    normalize_word_cav,
)
from src.lib.embeddings import batch_image_to_embeddings, get_image_paths, image_to_embedding
from src.lib.faiss_db import ImageVectorDB
from src.lib.models import setup_model

__all__ = [
    "setup_model",
    "get_image_paths",
    "image_to_embedding",
    "batch_image_to_embeddings",
    "compute_cav",
    "compute_cav_from_images",
    "compute_word_cav",
    "get_image_similarity_scores",
    "get_word_similarity_score",
    "normalize_cav",
    "normalize_word_cav",
    "ImageVectorDB",
]

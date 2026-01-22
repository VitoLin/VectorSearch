"""
Library modules for image similarity search.
"""

from .cav import (
    compute_cav,
    compute_cav_from_images,
    compute_word_cav,
    get_image_similarity_scores,
    get_word_similarity_score,
    normalize_cav,
    normalize_word_cav,
)
from .embeddings import batch_image_to_embeddings, get_image_paths, image_to_embedding
from .faiss_db import ImageVectorDB
from .models import setup_model

__all__ = [
    "compute_cav",
    "compute_cav_from_images",
    "compute_word_cav",
    "get_image_similarity_scores",
    "get_word_similarity_score",
    "normalize_cav",
    "normalize_word_cav",
    "get_image_paths",
    "image_to_embedding",
    "batch_image_to_embeddings",
    "ImageVectorDB",
    "setup_model",
]

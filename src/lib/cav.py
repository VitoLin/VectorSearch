#!/usr/bin/env python3
"""
Consolidated Concept Activation Vector (CAV) utilities.

This module provides tools for computing CAVs from both images and words.
"""

import numpy as np
from typing import List, Callable, Optional


def normalize_cav(cav):
    """
    Normalize CAV vector for cosine similarity.

    Args:
        cav: numpy array of shape (dim,)

    Returns:
        Normalized CAV vector
    """
    return cav / (np.linalg.norm(cav) + 1e-12)


def compute_cav(positive_embeddings, negative_embeddings=None):
    """
    Compute Concept Activation Vector (CAV) from embeddings.

    Args:
        positive_embeddings: numpy array [N, dim] - embeddings with the concept
        negative_embeddings: numpy array [M, dim] or None - embeddings without the concept
                         If None, uses zero vector

    Returns:
        CAV vector of shape (dim,)
    """
    # Compute mean of positive embeddings
    pos_mean = np.mean(positive_embeddings, axis=0)

    if negative_embeddings is not None and len(negative_embeddings) > 0:
        # Compute mean of negative embeddings and subtract
        neg_mean = np.mean(negative_embeddings, axis=0)
        cav = pos_mean - neg_mean
    else:
        # If no negatives, CAV is just the mean of positives
        cav = pos_mean

    print(f"  CAV dimension: {cav.shape}")
    print(f"  CAV norm before normalization: {np.linalg.norm(cav):.6f}")

    return cav


def get_similarity_scores(cav, embeddings):
    """
    Score embeddings based on their similarity to a CAV.

    Higher score = more aligned with CAV concept.

    Args:
        cav: CAV vector (dim,)
        embeddings: numpy array [N, dim] - embeddings to score

    Returns:
        numpy array [N] - similarity scores
    """
    scores = np.dot(embeddings, cav)
    return scores


def vectorize_words(model, words: List[str]) -> np.ndarray:
    """
    Convert words to embeddings using a sentence transformer model.

    Args:
        model: Sentence transformer model with .encode() method
        words: List of words to vectorize

    Returns:
        numpy array of shape (len(words), embedding_dim)
    """
    return model.encode(words, convert_to_numpy=True)


def vectorize_images(image_paths: List[str], image_to_embedding_fn: Callable) -> np.ndarray:
    """
    Convert image paths to embeddings using a provided embedding function.

    Args:
        image_paths: List of paths to images
        image_to_embedding_fn: Function that takes an image path and returns an embedding

    Returns:
        numpy array of shape (len(image_paths), embedding_dim)
    """
    embeddings = [image_to_embedding_fn(p) for p in image_paths]
    return np.array(embeddings)


# Backward compatibility wrappers

def compute_cav_from_images(positive_image_paths, negative_image_paths, image_to_embedding_fn):
    """
    Compute CAV directly from image paths.

    Args:
        positive_image_paths: List of paths to images with the concept
        negative_image_paths: List of paths to images without the concept
        image_to_embedding_fn: Function to convert image path to embedding

    Returns:
        CAV vector
    """
    print(f"\nComputing CAV from {len(positive_image_paths)} positive images...")
    if negative_image_paths:
        print(f"And {len(negative_image_paths)} negative images...")

    pos_embs = vectorize_images(positive_image_paths, image_to_embedding_fn)

    neg_embs = None
    if negative_image_paths:
        neg_embs = vectorize_images(negative_image_paths, image_to_embedding_fn)

    return compute_cav(pos_embs, neg_embs)


def compute_word_cav(model, pos_words: List[str], neg_words: List[str]):
    """
    Compute CAV from word lists using a sentence transformer model.

    Args:
        model: Sentence transformer model with .encode() method
        pos_words: List of positive words (words with the concept)
        neg_words: List of negative words (words without the concept)

    Returns:
        CAV vector
    """
    pos_emb = vectorize_words(model, pos_words)
    neg_emb = vectorize_words(model, neg_words)
    return compute_cav(pos_emb, neg_emb)


def get_image_similarity_scores(cav, image_embeddings):
    """
    Score images based on their similarity to a CAV.

    Higher score = more aligned with CAV concept.

    Args:
        cav: CAV vector (dim,)
        image_embeddings: numpy array [N, dim] - embeddings of images to score

    Returns:
        numpy array [N] - similarity scores
    """
    return get_similarity_scores(cav, image_embeddings)


def get_word_similarity_score(model, cav, words: List[str]):
    """
    Score words based on their similarity to a CAV.

    Higher score = more aligned with CAV concept.

    Args:
        model: Sentence transformer model with .encode() method
        cav: CAV vector (dim,)
        words: List of words to score

    Returns:
        numpy array [N] - similarity scores
    """
    word_emb = vectorize_words(model, words)
    return get_similarity_scores(cav, word_emb)


# Backward compatibility alias
normalize_word_cav = normalize_cav

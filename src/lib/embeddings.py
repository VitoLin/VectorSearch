#!/usr/bin/env python3
"""
Image to embedding conversion with multi-format support.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif")


def get_image_paths(folder_path):
    """
    Get all image file paths from a folder with supported formats.

    Args:
        folder_path: Path to folder containing images

    Returns:
        List of Path objects for valid images
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    images = [p for p in folder.glob("*") if p.suffix.lower() in SUPPORTED_FORMATS]

    print(f"Found {len(images)} images in {folder_path}")
    return sorted(images)


def image_to_embedding(image_path, model, preprocess, device):
    """
    Convert an image file to a normalized L2 embedding.

    Args:
        image_path: Path to image file (str or Path)
        model: PyTorch model
        preprocess: torchvision transforms pipeline
        device: torch.device

    Returns:
        numpy array: L2-normalized embedding vector
    """
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(x)

    feat = feat.cpu().numpy().astype("float32").squeeze(0)
    feat /= np.linalg.norm(feat) + 1e-10  # L2-normalize for cosine similarity

    return feat


def batch_image_to_embeddings(image_paths, model, preprocess, device, batch_size=32):
    """
    Convert multiple images to embeddings in batches.

    Args:
        image_paths: List of image paths
        model: PyTorch model
        preprocess: torchvision transforms pipeline
        device: torch.device
        batch_size: Number of images to process at once

    Returns:
        numpy array: Stack of embeddings [N, dim]
    """
    all_embs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_embs = [image_to_embedding(p, model, preprocess, device) for p in batch_paths]
        all_embs.extend(batch_embs)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {len(all_embs)}/{len(image_paths)} images...")

    return np.array(all_embs, dtype="float32")

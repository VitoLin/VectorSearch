#!/usr/bin/env python3
"""
Concept Activation Vectors (CAVs) for Images

This script demonstrates computing and using CAVs to measure how similar
images are to a concept (e.g., "dog-ness" vs cats).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import faiss


def setup_model(device=None):
    """Load pretrained MobileNetV2 and return model + preprocessing pipeline."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.mobilenet_v2(pretrained=True)

    model.classifier = torch.nn.Identity()  # type: ignore
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess, device


def image_to_embedding(path, model, preprocess, device):
    """Convert an image file to a normalized L2 embedding."""
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
    feat = feat.cpu().numpy().astype('float32').squeeze(0)
    feat /= np.linalg.norm(feat) + 1e-10
    return feat


def normalize_cav(cav):
    """Normalize CAV vector for cosine similarity."""
    return cav / (np.linalg.norm(cav) + 1e-12)


def compute_cav_images(model, preprocess, device, pos_image_paths, neg_image_paths):
    """
    Compute Concept Activation Vector (CAV) from positive and negative images.
    CAV = mean(positive_embeddings) - mean(negative_embeddings)
    """
    pos_embs = [image_to_embedding(path, model, preprocess, device) for path in pos_image_paths]
    neg_embs = [image_to_embedding(path, model, preprocess, device) for path in neg_image_paths]

    cav = np.mean(pos_embs, axis=0) - np.mean(neg_embs, axis=0)
    return cav


def get_image_similarity_scores(cav, image_paths, model, preprocess, device):
    """
    Score images based on their similarity to a CAV.
    Higher score = more aligned with CAV concept.
    """
    image_embs = [image_to_embedding(path, model, preprocess, device) for path in image_paths]
    scores = [np.dot(emb, cav) for emb in image_embs]
    return np.array(scores)


def build_faiss_index(image_paths, model, preprocess, device, dim=1280):
    """Build FAISS index for fast similarity search."""
    index = faiss.IndexFlatIP(dim)  # type: ignore

    image_embs = np.array([image_to_embedding(path, model, preprocess, device) for path in image_paths], dtype='float32')
    index.add(image_embs)  # type: ignore

    return index


def main():
    model, preprocess, device = setup_model()

    positive_images = ["./dog.jpg", "./dog2.jpg"]
    negative_images = ["./cat.jpg"]

    print("\n" + "=" * 60)
    print("Computing CAV for 'dog-ness' concept...")
    print("=" * 60)
    cav = compute_cav_images(model, preprocess, device, positive_images, negative_images)
    cav_norm = normalize_cav(cav)

    print(f"CAV dimension: {cav.shape}")
    print(f"CAV norm before normalization: {np.linalg.norm(cav):.6f}")
    print(f"CAV norm after normalization: {np.linalg.norm(cav_norm):.6f}")

    all_images = positive_images + negative_images
    scores = get_image_similarity_scores(cav_norm, all_images, model, preprocess, device)

    paired = list(zip(all_images, scores))
    paired.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("Images ranked by similarity to 'dog-ness' CAV:")
    print("=" * 60)
    for i, (img_path, score) in enumerate(paired, 1):
        print(f"{i}. {img_path:20s} -> {score:8.6f}")

    print("\n" + "=" * 60)
    print("Building FAISS index...")
    print("=" * 60)
    dim = 1280
    index = build_faiss_index(all_images, model, preprocess, device, dim)
    print(f"FAISS index built with {index.ntotal} vectors (dim={dim})")

    query_vec = cav_norm.reshape(1, -1).astype('float32')
    k = len(all_images)
    distances, indices = index.search(query_vec, k)  # type: ignore

    print("\n" + "=" * 60)
    print("Top images according to FAISS search:")
    print("=" * 60)
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        img_path = all_images[idx]
        print(f"{i}. {img_path:20s} -> {dist:8.6f}")

    print("\n" + "=" * 60)
    print("SUCCESS: All image CAV computations completed without errors!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Model loading and setup for image embeddings.
Supports MobileNetV2 (from existing examples).
"""

import os
import torch
from torchvision import transforms, models


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def setup_model(device=None, model_name='mobilenet_v2'):
    """
    Load pretrained model and return model + preprocessing pipeline.

    Args:
        device: torch.device or None (auto-detect)
        model_name: 'mobilenet_v2' (default)

    Returns:
        model, preprocess, device
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load MobileNetV2 (following existing example pattern)
    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.mobilenet_v2(pretrained=True)

    # Remove classifier to get feature vectors (1280-d)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)

    # Preprocessing pipeline (same as model expects)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return model, preprocess, device

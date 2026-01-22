"""
Tests for embeddings module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.lib.embeddings import (
    batch_image_to_embeddings,
    get_image_paths,
    image_to_embedding,
    SUPPORTED_FORMATS,
)


class TestSupportedFormats:
    """Test SUPPORTED_FORMATS constant."""

    def test_supported_formats_includes_common_formats(self):
        expected = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif']
        for fmt in expected:
            assert fmt in SUPPORTED_FORMATS


class TestGetImagePaths:
    """Test get_image_paths function."""

    def test_get_image_paths_empty_folder(self, tmp_path):
        folder = tmp_path / "images"
        folder.mkdir()
        paths = get_image_paths(folder)
        assert paths == []

    def test_get_image_paths_with_images(self, tmp_path):
        folder = tmp_path / "images"
        folder.mkdir()

        img1 = folder / "test1.jpg"
        img2 = folder / "test2.png"

        Image.new("RGB", (10, 10)).save(img1)
        Image.new("RGB", (10, 10)).save(img2)

        paths = get_image_paths(folder)
        assert len(paths) == 2
        assert img1 in paths
        assert img2 in paths

    def test_get_image_paths_filters_non_images(self, tmp_path):
        folder = tmp_path / "images"
        folder.mkdir()

        img = folder / "test.jpg"
        txt = folder / "test.txt"
        pdf = folder / "test.pdf"

        Image.new("RGB", (10, 10)).save(img)
        txt.write_text("test")
        pdf.write_text("test")

        paths = get_image_paths(folder)
        assert len(paths) == 1
        assert img in paths

    def test_get_image_paths_filters_unsupported_formats(self, tmp_path):
        folder = tmp_path / "images"
        folder.mkdir()

        img_jpg = folder / "test.jpg"
        img_png = folder / "test.png"
        img_xyz = folder / "test.xyz"

        Image.new("RGB", (10, 10)).save(img_jpg)
        Image.new("RGB", (10, 10)).save(img_png)
        img_xyz.write_text("test")

        paths = get_image_paths(folder)
        assert len(paths) == 2
        assert img_jpg in paths
        assert img_png in paths
        assert img_xyz not in paths

    def test_get_image_paths_case_insensitive(self, tmp_path):
        folder = tmp_path / "images"
        folder.mkdir()

        img1 = folder / "test.JPG"
        img2 = folder / "test.PNG"

        Image.new("RGB", (10, 10)).save(img1)
        Image.new("RGB", (10, 10)).save(img2)

        paths = get_image_paths(folder)
        assert len(paths) == 2

    def test_get_image_paths_sorted(self, tmp_path):
        folder = tmp_path / "images"
        folder.mkdir()

        img3 = folder / "c.jpg"
        img1 = folder / "a.jpg"
        img2 = folder / "b.jpg"

        Image.new("RGB", (10, 10)).save(img1)
        Image.new("RGB", (10, 10)).save(img2)
        Image.new("RGB", (10, 10)).save(img3)

        paths = get_image_paths(folder)
        assert paths[0] == img1
        assert paths[1] == img2
        assert paths[2] == img3

    def test_get_image_paths_nonexistent_folder(self, tmp_path):
        folder = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            get_image_paths(folder)


class TestImageToEmbedding:
    """Test image_to_embedding function."""

    @pytest.fixture
    def model_and_preprocess(self):
        """Create a simple mock model and preprocess."""
        import torch
        from torchvision import transforms

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(224 * 224 * 3, 1280)

            def forward(self, x):
                batch_size = x.shape[0]
                x = x.view(batch_size, -1)
                return self.layer(x)

        model = MockModel()
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        device = torch.device("cpu")

        return model, preprocess, device

    def test_image_to_embedding_returns_correct_shape(self, tmp_path, model_and_preprocess):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224)).save(img_path)

        model, preprocess, device = model_and_preprocess
        emb = image_to_embedding(img_path, model, preprocess, device)

        assert emb.shape == (1280,)
        assert emb.dtype == np.float32

    def test_image_to_embedding_normalizes(self, tmp_path, model_and_preprocess):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224)).save(img_path)

        model, preprocess, device = model_and_preprocess
        emb = image_to_embedding(img_path, model, preprocess, device)

        norm = np.linalg.norm(emb)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_image_to_embedding_supports_rgb_conversion(self, tmp_path, model_and_preprocess):
        img_path = tmp_path / "test.png"
        Image.new("L", (224, 224)).save(img_path)  # Grayscale image

        model, preprocess, device = model_and_preprocess
        emb = image_to_embedding(img_path, model, preprocess, device)

        assert emb.shape == (1280,)

    def test_image_to_embedding_string_path(self, tmp_path, model_and_preprocess):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224)).save(img_path)

        model, preprocess, device = model_and_preprocess
        emb = image_to_embedding(str(img_path), model, preprocess, device)

        assert emb.shape == (1280,)


class TestBatchImageToEmbeddings:
    """Test batch_image_to_embeddings function."""

    @pytest.fixture
    def model_and_preprocess(self):
        """Create a simple mock model and preprocess."""
        import torch
        from torchvision import transforms

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(224 * 224 * 3, 1280)

            def forward(self, x):
                batch_size = x.shape[0]
                x = x.view(batch_size, -1)
                return self.layer(x)

        model = MockModel()
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        device = torch.device("cpu")

        return model, preprocess, device

    def test_batch_image_to_embeddings_empty_list(self, model_and_preprocess):
        model, preprocess, device = model_and_preprocess
        embs = batch_image_to_embeddings([], model, preprocess, device)

        assert embs.shape == (0,)
        assert embs.dtype == np.float32

    def test_batch_image_to_embeddings_single_image(self, tmp_path, model_and_preprocess):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224)).save(img_path)

        model, preprocess, device = model_and_preprocess
        embs = batch_image_to_embeddings([img_path], model, preprocess, device)

        assert embs.shape == (1, 1280)
        assert embs.dtype == np.float32

    def test_batch_image_to_embeddings_multiple_images(self, tmp_path, model_and_preprocess):
        img1 = tmp_path / "test1.jpg"
        img2 = tmp_path / "test2.jpg"
        img3 = tmp_path / "test3.jpg"

        Image.new("RGB", (224, 224)).save(img1)
        Image.new("RGB", (224, 224)).save(img2)
        Image.new("RGB", (224, 224)).save(img3)

        model, preprocess, device = model_and_preprocess
        embs = batch_image_to_embeddings([img1, img2, img3], model, preprocess, device)

        assert embs.shape == (3, 1280)
        assert embs.dtype == np.float32

    def test_batch_image_to_embeddings_normalizes(self, tmp_path, model_and_preprocess):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224)).save(img_path)

        model, preprocess, device = model_and_preprocess
        embs = batch_image_to_embeddings([img_path], model, preprocess, device)

        norm = np.linalg.norm(embs[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_batch_image_to_embeddings_respects_batch_size(self, tmp_path, model_and_preprocess):
        for _ in range(10):
            img_path = tmp_path / f"test{_:02d}.jpg"
            Image.new("RGB", (224, 224)).save(img_path)

        model, preprocess, device = model_and_preprocess
        embs = batch_image_to_embeddings(
            list(tmp_path.glob("*.jpg")), model, preprocess, device, batch_size=3
        )

        assert embs.shape == (10, 1280)

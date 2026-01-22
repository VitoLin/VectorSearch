"""
Tests for CAV (Concept Activation Vector) module.
"""

import numpy as np

from src.lib.cav import (
    compute_cav,
    compute_cav_from_images,
    compute_word_cav,
    get_image_similarity_scores,
    get_word_similarity_score,
    normalize_cav,
    normalize_word_cav,
)


class MockModel:
    """Mock sentence transformer model for testing."""

    def encode(self, words, convert_to_numpy=True):
        dim = 10
        return np.array([np.random.rand(dim) for _ in words])


class TestNormalizeCav:
    """Test normalize_cav function."""

    def test_normalize_nonzero_vector(self):
        cav = np.array([3.0, 4.0])
        normalized = normalize_cav(cav)
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_unit_vector(self):
        cav = np.array([0.6, 0.8])
        normalized = normalize_cav(cav)
        assert np.allclose(normalized, cav)

    def test_normalize_unit_norm(self):
        cav = np.array([1.0, 2.0, 3.0])
        normalized = normalize_cav(cav)
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0)

    def test_normalize_high_dimensional(self):
        cav = np.random.rand(100)
        normalized = normalize_cav(cav)
        assert np.isclose(np.linalg.norm(normalized), 1.0)


class TestComputeCav:
    """Test compute_cav function."""

    def test_compute_cav_with_negatives(self):
        pos_embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        neg_embs = np.array([[-1.0, 0.0], [0.0, -1.0]])
        cav = compute_cav(pos_embs, neg_embs)
        expected = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(cav, expected)

    def test_compute_cav_without_negatives(self):
        pos_embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        cav = compute_cav(pos_embs)
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(cav, expected)

    def test_compute_cav_empty_negatives(self):
        pos_embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        neg_embs = np.array([])
        cav = compute_cav(pos_embs, neg_embs)
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(cav, expected)

    def test_compute_cav_returns_correct_shape(self):
        pos_embs = np.random.rand(5, 10)
        neg_embs = np.random.rand(3, 10)
        cav = compute_cav(pos_embs, neg_embs)
        assert cav.shape == (10,)

    def test_compute_cav_single_positive(self):
        pos_embs = np.array([[1.0, 2.0, 3.0]])
        neg_embs = np.array([[0.0, 0.0, 0.0]])
        cav = compute_cav(pos_embs, neg_embs)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(cav, expected)


class TestComputeCavFromImages:
    """Test compute_cav_from_images function."""

    def test_compute_cav_from_images_with_negatives(self):
        pos_paths = ["img1.jpg", "img2.jpg"]
        neg_paths = ["img3.jpg"]
        mock_fn = (
            lambda p: np.array([1.0, 0.0]) if "img1" in p or "img2" in p else np.array([-1.0, 0.0])
        )
        cav = compute_cav_from_images(pos_paths, neg_paths, mock_fn)
        assert cav.shape == (2,)

    def test_compute_cav_from_images_without_negatives(self):
        pos_paths = ["img1.jpg", "img2.jpg"]
        mock_fn = lambda p: np.array([1.0, 0.0])
        cav = compute_cav_from_images(pos_paths, [], mock_fn)
        assert cav.shape == (2,)

    def test_compute_cav_from_images_none_negatives(self):
        pos_paths = ["img1.jpg"]
        mock_fn = lambda p: np.array([1.0, 0.0])
        cav = compute_cav_from_images(pos_paths, None, mock_fn)
        assert cav.shape == (2,)


class TestComputeWordCav:
    """Test compute_word_cav function."""

    def test_compute_word_cav_basic(self):
        model = MockModel()
        pos_words = ["happy", "joy"]
        neg_words = ["sad", "gloomy"]
        cav = compute_word_cav(model, pos_words, neg_words)
        assert cav.shape == (10,)

    def test_compute_word_cav_returns_correct_shape(self):
        model = MockModel()
        pos_words = ["test1", "test2", "test3"]
        neg_words = ["test4", "test5"]
        cav = compute_word_cav(model, pos_words, neg_words)
        assert cav.shape == (10,)


class TestGetImageSimilarityScores:
    """Test get_image_similarity_scores function."""

    def test_get_image_similarity_scores_basic(self):
        cav = np.array([1.0, 0.0])
        image_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        scores = get_image_similarity_scores(cav, image_embeddings)
        np.testing.assert_array_almost_equal(scores, [1.0, 0.0])

    def test_get_image_similarity_scores_returns_correct_shape(self):
        cav = np.random.rand(10)
        image_embeddings = np.random.rand(5, 10)
        scores = get_image_similarity_scores(cav, image_embeddings)
        assert scores.shape == (5,)


class TestGetWordSimilarityScore:
    """Test get_word_similarity_score function."""

    def test_get_word_similarity_score_basic(self):
        model = MockModel()
        cav = np.array([1.0, 0.0, 0.0] + [0.0] * 7)
        words = ["test1", "test2"]
        scores = get_word_similarity_score(model, cav, words)
        assert scores.shape == (2,)

    def test_get_word_similarity_score_returns_correct_shape(self):
        model = MockModel()
        cav = np.random.rand(10)
        words = ["a", "b", "c"]
        scores = get_word_similarity_score(model, cav, words)
        assert scores.shape == (3,)


class TestNormalizeWordCav:
    """Test normalize_word_cav alias function."""

    def test_normalize_word_cav_alias(self):
        cav = np.array([3.0, 4.0])
        normalized = normalize_word_cav(cav)
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_word_cav_same_as_normalize_cav(self):
        cav = np.random.rand(10)
        norm1 = normalize_cav(cav)
        norm2 = normalize_word_cav(cav)
        np.testing.assert_array_almost_equal(norm1, norm2)

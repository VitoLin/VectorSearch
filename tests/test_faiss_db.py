"""
Tests for FAISS vector database module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.lib.faiss_db import ImageVectorDB


class TestImageVectorDBInit:
    """Test ImageVectorDB initialization."""

    def test_init_without_dim(self):
        db = ImageVectorDB()
        assert db.index is None
        assert db.image_paths == []
        assert db.dim is None

    def test_init_with_dim(self):
        db = ImageVectorDB(dim=1280)
        assert db.index is None
        assert db.image_paths == []
        assert db.dim == 1280


class TestImageVectorDBBuildIndex:
    """Test ImageVectorDB.build_index method."""

    def test_build_index_with_ip_metric(self):
        db = ImageVectorDB()
        embeddings = np.random.rand(5, 10).astype('float32')
        image_paths = [f"img{i}.jpg" for i in range(5)]

        index = db.build_index(embeddings, image_paths, metric='ip')

        assert index is not None
        assert index.ntotal == 5
        assert db.dim == 10
        assert len(db.image_paths) == 5

    def test_build_index_with_l2_metric(self):
        db = ImageVectorDB()
        embeddings = np.random.rand(5, 10).astype('float32')
        image_paths = [f"img{i}.jpg" for i in range(5)]

        index = db.build_index(embeddings, image_paths, metric='l2')

        assert index is not None
        assert index.ntotal == 5
        assert db.dim == 10

    def test_build_index_invalid_metric(self):
        db = ImageVectorDB()
        embeddings = np.random.rand(5, 10).astype('float32')
        image_paths = [f"img{i}.jpg" for i in range(5)]

        with pytest.raises(ValueError, match="Unknown metric"):
            db.build_index(embeddings, image_paths, metric='invalid')

    def test_build_index_converts_paths_to_strings(self):
        db = ImageVectorDB()
        embeddings = np.random.rand(3, 10).astype('float32')
        image_paths = [Path(f"img{i}.jpg") for i in range(3)]

        db.build_index(embeddings, image_paths, metric='ip')

        for path in db.image_paths:
            assert isinstance(path, str)

    def test_build_index_sets_dimension(self):
        db = ImageVectorDB()
        embeddings = np.random.rand(5, 20).astype('float32')
        image_paths = [f"img{i}.jpg" for i in range(5)]

        db.build_index(embeddings, image_paths, metric='ip')

        assert db.dim == 20

    def test_build_index_empty_embeddings(self):
        db = ImageVectorDB()
        embeddings = np.array([]).reshape(0, 10).astype('float32')
        image_paths = []

        db.build_index(embeddings, image_paths, metric='ip')

        assert db.index.ntotal == 0


class TestImageVectorDBSearch:
    """Test ImageVectorDB.search method."""

    @pytest.fixture
    def populated_db(self):
        """Create a populated database for testing."""
        db = ImageVectorDB()

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ], dtype='float32')

        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]

        db.build_index(embeddings, image_paths, metric='ip')
        return db

    def test_search_without_building_index(self):
        db = ImageVectorDB()
        query = np.array([1.0, 0.0, 0.0])

        with pytest.raises(RuntimeError, match="Index not built"):
            db.search(query, k=3)

    def test_search_returns_correct_number_of_results(self, populated_db):
        query = np.array([1.0, 0.0, 0.0])
        results = populated_db.search(query, k=2)

        assert len(results) == 2

    def test_search_returns_k_results_if_database_large_enough(self, populated_db):
        query = np.array([1.0, 0.0, 0.0])
        results = populated_db.search(query, k=4)

        assert len(results) == 4

    def test_search_limits_to_database_size(self, populated_db):
        query = np.array([1.0, 0.0, 0.0])
        results = populated_db.search(query, k=10)

        assert len(results) == 4  # Database only has 4 items

    def test_search_returns_tuples_of_path_and_score(self, populated_db):
        query = np.array([1.0, 0.0, 0.0])
        results = populated_db.search(query, k=2)

        for path, score in results:
            assert isinstance(path, str)
            assert isinstance(score, float)

    def test_search_1d_query_vector(self, populated_db):
        query = np.array([1.0, 0.0, 0.0])
        results = populated_db.search(query, k=1)

        assert len(results) == 1

    def test_search_2d_query_vector(self, populated_db):
        query = np.array([[1.0, 0.0, 0.0]])
        results = populated_db.search(query, k=1)

        assert len(results) == 1

    def test_search_ip_metric_highest_similarity_first(self, populated_db):
        query = np.array([1.0, 0.0, 0.0])
        results = populated_db.search(query, k=4)

        # First result should be img1.jpg which is closest to [1, 0, 0]
        assert results[0][0] == "img1.jpg"
        # For IP metric with normalized vectors, highest score = most similar
        assert results[0][1] >= results[1][1]


class TestImageVectorDBSave:
    """Test ImageVectorDB.save method."""

    def test_save_without_building_index(self, tmp_path):
        db = ImageVectorDB()
        prefix = tmp_path / "test_db"

        with pytest.raises(RuntimeError, match="No index to save"):
            db.save(prefix)

    def test_save_creates_files(self, tmp_path):
        db = ImageVectorDB()
        embeddings = np.random.rand(3, 10).astype('float32')
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

        db.build_index(embeddings, image_paths, metric='ip')

        prefix = tmp_path / "test_db"
        db.save(prefix)

        assert Path(f"{prefix}.faiss").exists()
        assert Path(f"{prefix}_paths.pkl").exists()

    def test_save_with_str_prefix(self, tmp_path):
        db = ImageVectorDB()
        embeddings = np.random.rand(3, 10).astype('float32')
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

        db.build_index(embeddings, image_paths, metric='ip')

        prefix = str(tmp_path / "test_db")
        db.save(prefix)

        assert Path(f"{prefix}.faiss").exists()
        assert Path(f"{prefix}_paths.pkl").exists()


class TestImageVectorDBLoad:
    """Test ImageVectorDB.load method."""

    def test_load_creates_index(self, tmp_path):
        # First save a database
        db1 = ImageVectorDB()
        embeddings = np.random.rand(3, 10).astype('float32')
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

        db1.build_index(embeddings, image_paths, metric='ip')
        prefix = tmp_path / "test_db"
        db1.save(prefix)

        # Now load it
        db2 = ImageVectorDB()
        db2.load(prefix)

        assert db2.index is not None
        assert db2.index.ntotal == 3
        assert db2.dim == 10

    def test_load_restores_image_paths(self, tmp_path):
        # First save a database
        db1 = ImageVectorDB()
        embeddings = np.random.rand(3, 10).astype('float32')
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

        db1.build_index(embeddings, image_paths, metric='ip')
        prefix = tmp_path / "test_db"
        db1.save(prefix)

        # Now load it
        db2 = ImageVectorDB()
        db2.load(prefix)

        assert db2.image_paths == ["img1.jpg", "img2.jpg", "img3.jpg"]

    def test_load_nonexistent_files(self, tmp_path):
        db = ImageVectorDB()
        prefix = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Index files not found"):
            db.load(prefix)

    def test_load_missing_index_file(self, tmp_path):
        db = ImageVectorDB()
        prefix = tmp_path / "incomplete"

        # Create only paths file
        with open(f"{prefix}_paths.pkl", "wb") as f:
            import pickle
            pickle.dump([], f)

        with pytest.raises(FileNotFoundError):
            db.load(prefix)

    def test_load_missing_paths_file(self, tmp_path):
        db = ImageVectorDB()
        prefix = tmp_path / "incomplete"

        # Create only index file
        import faiss
        index = faiss.IndexFlatIP(10)
        faiss.write_index(index, f"{prefix}.faiss")

        with pytest.raises(FileNotFoundError):
            db.load(prefix)

    def test_roundtrip_save_and_load(self, tmp_path):
        # Create database and search
        db1 = ImageVectorDB()
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype='float32')
        image_paths = ["img1.jpg", "img2.jpg"]

        db1.build_index(embeddings, image_paths, metric='ip')
        original_results = db1.search(np.array([1.0, 0.0]), k=2)

        # Save and load
        prefix = tmp_path / "test_db"
        db1.save(prefix)

        db2 = ImageVectorDB()
        db2.load(prefix)
        loaded_results = db2.search(np.array([1.0, 0.0]), k=2)

        # Results should be identical
        assert len(original_results) == len(loaded_results)
        for (orig_path, orig_score), (load_path, load_score) in zip(original_results, loaded_results):
            assert orig_path == load_path
            assert np.isclose(orig_score, load_score)

# Image Similarity Search with CAV and FAISS

Search for similar images using Concept Activation Vectors (CAV) and FAISS vector database.

## Features

- **Multi-format support**: Handles JPG, PNG, WebP, BMP, TIFF, GIF images
- **CAV computation**: Extract concept vectors from positive and negative example images
- **FAISS indexing**: Fast similarity search across large image databases
- **Batch processing**: Efficiently process thousands of images
- **CLI interface**: Simple command-line tools for building and querying databases

## Setup

```bash
# Install dependencies (already in pyproject.toml)
pip install -e .
```

## Usage

### Build a FAISS Database

Index all images in a folder for fast similarity search:

```bash
python -m src.main build-db --folder ./database_images/ --output my_database
```

Options:
- `--folder`: Path to folder containing images (required)
- `--output`: Output prefix for index files (default: `image_db`)
- `--batch-size`: Batch size for embedding generation (default: `32`)

This creates:
- `my_database.faiss` - FAISS index
- `my_database_paths.pkl` - Image path mappings

### Search for Similar Images

#### Using a single query folder

```bash
python -m src.main search --query ./query_images/ --db my_database --k 20
```

#### Using positive and negative concepts (CAV)

```bash
# Dogs vs Cats
python -m src.main search \
  --query-positive ./dogs/ \
  --query-negative ./cats/ \
  --db my_database \
  --k 10
```

Options:
- `--query`: Folder with query images (simple search)
- `--query-positive`: Folder with positive example images (CAV)
- `--query-negative`: Folder with negative example images (CAV, optional)
- `--db`: Prefix of saved database (required)
- `--k`: Number of results to return (default: `10`)
- `--output`: Save results to file (optional)

## How It Works

1. **Image Embedding**: Uses MobileNetV2 to convert images to 1280-d feature vectors
2. **CAV Computation**: `CAV = mean(positive_embeddings) - mean(negative_embeddings)`
3. **FAISS Index**: Stores embeddings in vector database for fast search
4. **Similarity Search**: Finds images with highest cosine similarity to CAV

## Example Workflow

```bash
# 1. Build database from animal images folder
python -m src.main build-db --folder animals/ --output animal_db

# 2. Search for "dog-like" images
python -m src.main search \
  --query-positive dog_samples/ \
  --query-negative cat_samples/ \
  --db animal_db \
  --k 5
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff)
- GIF (.gif)

## Model Details

- **Base Model**: MobileNetV2 (ImageNet pretrained)
- **Feature Dimension**: 1280
- **Normalization**: L2-normalized for cosine similarity
- **Index Type**: FAISS IndexFlatIP (Inner Product)

## Module Structure

```
src/
├── models.py       # Model loading and preprocessing
├── embeddings.py   # Image to embedding conversion
├── cav.py          # CAV computation utilities
├── faiss_db.py    # FAISS index operations
└── main.py        # CLI interface
```

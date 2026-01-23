# VectorSearch

Vector search toolkit with Concept Activation Vectors (CAV) and FAISS for image and word similarity searches.

## Quick Start

```bash
# Install dependencies
uv sync

# Image search - Build database
uv run python -m src.main build-db --folder ./images/ --output my_db

# Image search - Find similar images (CAV)
uv run python -m src.main search --query-positive ./dogs/ --query-negative ./cats/ --db my_db --k 10

# Word search - Build index
uv run python vector_db.py build --out-prefix word_index

# Word search - Query words
uv run python vector_db.py query --word king --out-prefix word_index --k 5
```

## Features

- **Image Similarity**: MobileNetV2 embeddings + CAV for semantic image search
- **Word Similarity**: Word2Vec + FAISS for semantic word search
- **FAISS Indexing**: Fast similarity search across large datasets
- **Docker Support**: Complete containerized environment

## Image Search

### Build Database
```bash
uv run python -m src.main build-db --folder ./images/ --output my_db --batch-size 32
```
Creates `my_db.faiss` (FAISS index) and `my_db_paths.pkl` (image paths)

### Search
```bash
# Using positive/negative examples (CAV)
uv run python -m src.main search --query-positive ./dogs/ --query-negative ./cats/ --db my_db --k 10

# Using single query folder
uv run python -m src.main search --query ./query_images/ --db my_db --k 20
```

## Word Search

### Build Index
```bash
uv run python vector_db.py build --out-prefix word_index --vector-size 100 --epochs 50
```

### Query
```bash
uv run python vector_db.py query --word king --out-prefix word_index --k 5
```

## How It Works

**Image Search**: `CAV = mean(positive_embeddings) - mean(negative_embeddings)` → Search FAISS index for highest cosine similarity

**Word Search**: Word2Vec embeddings → FAISS IndexFlatL2 → Find nearest neighbors by Euclidean distance

## Supported Image Formats

JPEG, PNG, WebP, BMP, TIFF, GIF

## Module Structure

```
vectorsearch/
├── src/lib/          # Image search utilities (models, embeddings, CAV, FAISS)
├── src/main.py       # Image search CLI
├── vector_db.py      # Word search CLI
├── tests/            # Test suite
├── examples/         # Jupyter notebooks
└── pyproject.toml    # Dependencies and config
```

## Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Type check
uv run mypy .
```

## Docker

```bash
# Build and run
./docker.sh build
./docker.sh build-db --folder /app/data/images --output my_db
./docker.sh search --query-positive /app/data/dogs --db my_db --k 10

# Word search
./docker.sh word-build --out-prefix word_index
./docker.sh word-query --word king --out-prefix word_index --k 5
```

See [DOCKER.md](DOCKER.md) for complete Docker guide.

## Troubleshooting

**"No module named 'src'"**: Run from project root with `uv run python -m src.main`

**Out of memory**: Reduce batch size: `--batch-size 16`

**Device selection**: Automatically detects MPS (Apple Silicon), CUDA, or CPU

## Use Cases

**Image**: Similar animals, style filtering (day/night), scene classification (indoor/outdoor), emotion detection

**Word**: Semantic similarity, word analogies, thematic clustering, vocabulary expansion

## Resources

- [DOCKER.md](DOCKER.md) - Docker setup guide
- [AGENTS.md](AGENTS.md) - Development guidelines
- [USAGE.md](USAGE.md) - Additional examples

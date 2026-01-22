# AGENTS.md - VectorSearch Development Guide

This file provides guidelines and commands for agents working on the VectorSearch project.

## Project Overview

VectorSearch is a Python project implementing word embeddings and vector database functionality using FAISS (Facebook AI Similarity Search) and Word2Vec. The project uses `uv` as its package manager and requires Python 3.11+.

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies from pyproject.toml
uv sync

# Add new dependencies
uv add <package-name>
```

### Running the Application

```bash
# Build the vector index
python vector_db.py build --out-prefix word_index

# Query nearest neighbors for a word
python vector_db.py query --word king --out-prefix word_index --k 5

# Build with custom vector size and epochs
python vector_db.py build --out-prefix word_index --vector-size 100 --epochs 50
```

### Testing

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_vector_db.py

# Run a single test function
pytest tests/test_vector_db.py::test_function_name

# Run with verbose output
pytest -v
```

### Linting and Type Checking

This project uses **ruff** for lint/formatting and **mypy** for type checking.

```bash
# Check linting (ruff)
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Type check (mypy)
uv run mypy .

# Format code
uv run ruff format .
```

**Configuration:** Ruff and mypy settings are in `pyproject.toml`.

## Code Style Guidelines

### Imports

Organize imports in the following order with blank lines between groups:

1. Standard library imports (`os`, `argparse`, `typing`, etc.)
2. Third-party imports (`numpy`, `faiss`, `gensim`, etc.)

Sort alphabetically within each group:

```python
import argparse
import os
import pickle
from typing import List

import numpy as np
from gensim.models import Word2Vec
import faiss
```

### Type Hints

Use type hints for all function signatures. Import types from `typing` module:

```python
def train_word2vec(corpus: List[str], vector_size: int = 100,
                   window: int = 5, min_count: int = 1,
                   epochs: int = 50, workers: int = 1) -> Word2Vec:
    ...
```

### Naming Conventions

- **Functions and variables:** snake_case (e.g., `train_word2vec`, `vector_size`)
- **Constants:** UPPER_CASE (e.g., `DEMO_CORPUS`, `DEFAULT_K`)
- **Classes:** PascalCase (not yet used in this project)

### Docstrings

Use Google-style docstrings with `Args` and `Returns` sections:

```python
def query(word: str, out_prefix: str, k: int = 5) -> None:
    """Query nearest neighbors for a given word.

    Args:
        word: The word to find neighbors for.
        out_prefix: Prefix used when building the index.
        k: Number of neighbors to return.

    Returns:
        None (prints results to stdout).
    """
```

Use module-level docstrings for main scripts with usage examples:

```python
"""
Simple Word2Vec + FAISS vector DB for word lookup.

Usage:
  Build index:
    python vector_db.py build --out-prefix word_index

  Query:
    python vector_db.py query --word king --out-prefix word_index --k 5
"""
```

### Formatting

- Use 4 spaces for indentation (no tabs)
- Limit line length to 100 characters
- Use f-strings for string formatting: `f"Saved FAISS index to {index_path}"`
- Add spaces around operators: `d = model.wv.vector_size`, not `d=model.wv.vector_size`
- Use spaces after commas: `model.wv.get_vector(word).astype('float32').reshape(1, -1)`

### Error Handling

Use early returns with descriptive error messages:

```python
def load_index_and_words(out_prefix: str):
    index_path = f"{out_prefix}.faiss"
    words_path = f"{out_prefix}_words.pkl"
    if not os.path.exists(index_path) or not os.path.exists(words_path):
        raise FileNotFoundError("Index or words file not found. Run build first.")
    ...
```

### CLI Design

Use `argparse` with subparsers for command-line interfaces:

```python
def main():
    parser = argparse.ArgumentParser(description="Word2Vec + FAISS toy vector DB")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Train Word2Vec and build FAISS index")
    p_build.add_argument("--out-prefix", default="word_index", help="Output prefix")

    p_query = sub.add_parser("query", help="Query nearest words")
    p_query.add_argument("--word", required=True, help="Word to query")

    args = parser.parse_args()
    ...
```

### Numerical Stability

When performing numerical operations, add small epsilon values to prevent division by zero:

```python
def normalize_cav(cav):
    return cav / (np.linalg.norm(cav) + 1e-12)
```

### File Organization

- Main entry points: root directory (`vector_db.py`)
- Library code: `src/lib/` directory
- Keep modules focused and small (under 200 lines preferred)
- Use absolute imports for consistency

## Project Structure

```
vectorsearch/
├── vector_db.py              # Main CLI tool
├── src/lib/cav.py            # CAV utilities
├── pyproject.toml            # Project configuration
├── .python-version           # Python version specification
├── README.md                 # Setup instructions
└── PLANNING.md               # Project planning notes
```

## Recommended Future Enhancements

Consider adding these tools for better code quality:

1. **pre-commit hooks** - To enforce code quality before commits

## Cursor Rules

No `.cursorrules` file exists. Create one at the project root if desired, referencing this AGENTS.md file.

## Copilot Instructions

No `.github/copilot-instructions.md` file exists. Create one if desired, referencing this AGENTS.md file.

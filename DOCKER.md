# Docker Setup for VectorSearch

This guide covers running VectorSearch in Docker containers.

## Prerequisites

- Docker Desktop (Mac/Windows) or Docker Engine (Linux)
- At least 4GB RAM available for Docker

## Quick Start

1. **Build the image:**
   ```bash
   ./docker.sh build
   ```

2. **Run a command:**
   ```bash
   ./docker.sh run-cmd python -m src.main --help
   ```

## Directory Structure

```
vectorsearch/
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Multi-container orchestration
├── docker.sh              # Helper script
├── data/                  # Input images (mounted to container)
├── outputs/               # Generated index files (mounted to container)
└── examples/              # Jupyter notebooks (for jupyter service)
```

## Helper Script Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `./docker.sh build` | Build Docker image |
| `./docker.sh up` | Start services in background |
| `./docker.sh down` | Stop all services |
| `./docker.sh run-cmd <args>` | Run any command in container |
| `./docker.sh build-db <args>` | Build FAISS database |
| `./docker.sh search <args>` | Search database |
| `./docker.sh word-build <args>` | Build word index |
| `./docker.sh word-query <args>` | Query word index |
| `./docker.sh jupyter` | Start Jupyter notebook |
| `./docker.sh shell` | Open bash shell in container |
| `./docker.sh help` | Show help message |

### Examples

#### Build Image Database

```bash
# Prepare your images in ./data folder
mkdir -p data/animals
cp your_images/*.jpg data/animals/

# Build database
./docker.sh build-db --folder /app/data/animals --output animals_db
```

#### Search Database

```bash
# Create query folders
mkdir -p data/dogs data/cats
cp dog_images/*.jpg data/dogs/
cp cat_images/*.jpg data/cats/

# Search
./docker.sh search --query-positive /app/data/dogs --query-negative /app/data/cats --db animals_db --k 10
```

#### Word Search (Word2Vec + FAISS)

```bash
# Build word index
./docker.sh word-build --out-prefix word_index

# Query words
./docker.sh word-query --word king --out-prefix word_index --k 5
```

#### Jupyter Notebook

```bash
# Start Jupyter (opens at http://localhost:8888)
./docker.sh jupyter
```

#### Interactive Shell

```bash
# Open shell in container
./docker.sh shell

# Now you can run any command
uv run python -m src.main build-db --folder /app/data/images --output my_db
```

## Direct Docker Commands

### Using docker-compose

```bash
# Build image
docker-compose build

# Run single command
docker-compose run --rm vectorsearch uv run python -m src.main --help

# Build database
docker-compose run --rm vectorsearch uv run python -m src.main build-db --folder /app/data/animals --output my_db

# Search
docker-compose run --rm vectorsearch uv run python -m src.main search --query-positive /app/data/dogs --db my_db --k 10

# Keep container running for multiple commands
docker-compose run --rm vectorsearch bash
```

### Using docker run directly

```bash
# Build image
docker build -t vectorsearch .

# Run with volume mounts
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  vectorsearch \
  uv run python -m src.main build-db --folder /app/data/animals --output my_db
```

## Container Architecture

### Base Image

- **Python**: 3.11-slim (minimal Alpine-based image)
- **Package Manager**: uv (fast Python package installer)
- **Dependencies**: All specified in pyproject.toml

### Services

#### vectorsearch (main service)

- **Purpose**: Run CLI commands for image and word search
- **Volumes**:
  - `./data` → `/app/data` (input images)
  - `./outputs` → `/app/outputs` (generated index files)
- **Command**: `python -m src.main --help` (default)

#### jupyter (optional service)

- **Purpose**: Run Jupyter notebooks for interactive analysis
- **Ports**: 8888 (http://localhost:8888)
- **Volumes**:
  - `./data` → `/app/data`
  - `./outputs` → `/app/outputs`
  - `./examples` → `/app/examples`

## Troubleshooting

### Image Build Fails

```bash
# Check Docker is running
docker ps

# Check available disk space
docker system df

# Clean up unused images
docker system prune -a
```

### Permission Issues on Linux

```bash
# Fix permission for mounted volumes
sudo chown -R $USER:$USER ./data ./outputs
```

### Out of Memory Errors

```bash
# Increase Docker memory limit in Docker Desktop
# Settings -> Resources -> Memory -> Set to 6GB or more
```

### Container Can't Find Files

```bash
# Verify volumes are mounted correctly
docker-compose run --rm vectorsearch ls -la /app/data

# Check container working directory
docker-compose run --rm vectorsearch pwd
```

### Jupyter Won't Start

```bash
# Check if port 8888 is already in use
lsof -i :8888

# Kill process using port 8888
kill -9 <PID>

# Or use a different port in docker-compose.yml
ports:
  - "8889:8888"
```

## Performance Tips

1. **Batch Size**: Increase `--batch-size` for faster embedding generation (default: 32)
   ```bash
   ./docker.sh build-db --folder /app/data/images --output my_db --batch-size 64
   ```

2. **Cache Docker Layers**: Dependencies are cached after first build
   ```bash
   # Only rebuild when pyproject.toml changes
   ./docker.sh build
   ```

3. **Use Outputs Volume**: Index files persist in `./outputs/` between runs

4. **Parallel Processing**: For large datasets, consider running multiple containers
   ```bash
   docker-compose up --scale vectorsearch=3
   ```

## Resource Requirements

| Operation | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Build image | Low | 1-2 GB | 2-3 GB |
| Build database (100 images) | Medium | 2-4 GB | 500 MB |
| Build database (10K images) | High | 8-16 GB | 50 GB |
| Search query | Low | 1-2 GB | <10 MB |
| Jupyter | Low | 2-4 GB | 1-2 GB |

## Advanced Usage

### Custom Dockerfile

To modify the container, create a custom Dockerfile:

```dockerfile
FROM vectorsearch

# Install additional packages
RUN uv add scikit-learn pandas

# Set environment variables
ENV CUSTOM_VAR=value
```

### Development Container

For development with live code reloading:

```bash
# Mount entire project directory
docker run --rm \
  -v $(pwd):/app \
  vectorsearch \
  uv run python -m src.main build-db --folder /app/data/images --output my_db
```

### GPU Support (Linux only)

If you have NVIDIA GPU, modify Dockerfile:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
# ... rest of Dockerfile
```

Then run with GPU:
```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  vectorsearch \
  uv run python -m src.main build-db --folder /app/data/images --output my_db
```

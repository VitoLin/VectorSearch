#!/bin/bash

# Helper script for running VectorSearch in Docker

set -e

show_help() {
    cat << EOF
VectorSearch Docker Helper

Usage:
    ./docker.sh <command> [options]

Commands:
    build           Build Docker image
    up              Start services in background
    down            Stop services
    run-cmd <args>  Run a command in the container
    build-db <args> Build FAISS database
    search <args>   Search the database
    jupyter         Start Jupyter notebook
    shell           Open shell in container
    help            Show this help message

Examples:
    ./docker.sh build
    ./docker.sh run-cmd python vector_db.py build --out-prefix word_index
    ./docker.sh build-db --folder ./data/animals --output my_db
    ./docker.sh search --query-positive ./data/dogs --db my_db --k 10
    ./docker.sh jupyter
    ./docker.sh shell

Note:
    - Data files should be in ./data folder (mounted to /app/data)
    - Output files will be saved to ./outputs folder (mounted to /app/outputs)
EOF
}

case "$1" in
    build)
        docker-compose build
        ;;
    up)
        docker-compose up -d
        echo "Services started. Use './docker.sh shell' to enter container."
        ;;
    down)
        docker-compose down
        ;;
    run-cmd)
        shift
        docker-compose run --rm vectorsearch uv run "$@"
        ;;
    build-db)
        shift
        docker-compose run --rm vectorsearch uv run python -m src.main build-db "$@"
        ;;
    search)
        shift
        docker-compose run --rm vectorsearch uv run python -m src.main search "$@"
        ;;
    word-query)
        shift
        docker-compose run --rm vectorsearch uv run python vector_db.py query "$@"
        ;;
    word-build)
        shift
        docker-compose run --rm vectorsearch uv run python vector_db.py build "$@"
        ;;
    jupyter)
        echo "Jupyter starting at http://localhost:8888"
        docker-compose up jupyter
        ;;
    shell)
        docker-compose run --rm vectorsearch /bin/bash
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './docker.sh help' for usage information."
        exit 1
        ;;
esac

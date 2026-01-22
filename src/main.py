#!/usr/bin/env python3
"""
Image similarity search using CAV and FAISS.

Usage:
    # Build FAISS index from folder of images
    python -m src.main build-db --folder database_images/ --output image_db

    # Search for similar images using CAV from query images
    python -m src.main search --query query_images/ --db image_db --k 20

    # Search with positive and negative concepts
    python -m src.main search --query-positive dogs/ --query-negative cats/ --db image_db --k 10
"""

import argparse

from src.lib.models import setup_model
from src.lib.embeddings import get_image_paths, image_to_embedding, batch_image_to_embeddings
from src.lib.cav import compute_cav_from_images, normalize_cav
from src.lib.faiss_db import ImageVectorDB


def cmd_build_db(args):
    """Build FAISS index from image folder."""
    print("=" * 60)
    print("Building FAISS Image Database")
    print("=" * 60)

    # Load model
    model, preprocess, device = setup_model()

    # Get all images
    image_paths = get_image_paths(args.folder)
    if len(image_paths) == 0:
        print("No images found in folder!")
        return

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = batch_image_to_embeddings(
        image_paths, model, preprocess, device, args.batch_size
    )

    # Build index
    db = ImageVectorDB()
    db.build_index(embeddings, image_paths, metric='ip')

    # Save
    db.save(args.output)

    print("\n" + "=" * 60)
    print("Database built successfully!")
    print("=" * 60)


def cmd_search(args):
    """Search for similar images using CAV."""
    print("=" * 60)
    print("Searching Image Database")
    print("=" * 60)

    # Load model
    model, preprocess, device = setup_model()

    # Load database
    print(f"\nLoading database from {args.db}...")
    db = ImageVectorDB()
    db.load(args.db)

    # Create embedding function for convenience
    def to_emb(path):
        return image_to_embedding(path, model, preprocess, device)

    # Compute CAV from query images
    print("\n" + "=" * 60)
    print("Computing CAV from query images...")
    print("=" * 60)

    if args.query_positive:
        # Use positive and negative folders
        pos_paths = get_image_paths(args.query_positive)
        neg_query_paths = get_image_paths(args.query_negative) if args.query_negative else []
        cav = compute_cav_from_images(pos_paths, neg_query_paths, to_emb)
    else:
        # Use single query folder
        query_paths = get_image_paths(args.query)
        neg_query_paths = get_image_paths(args.query_negative) if args.query_negative else []

        if neg_query_paths:
            cav = compute_cav_from_images(query_paths, neg_query_paths, to_emb)
        else:
            # No negative examples - just use mean of queries
            embs = [to_emb(p) for p in query_paths]
            cav = sum(embs) / len(embs)

    # Normalize CAV
    cav_norm = normalize_cav(cav)
    print(f"CAV norm after normalization: {cav_norm.shape}")

    # Search database
    print("\n" + "=" * 60)
    print(f"Searching top {args.k} most similar images...")
    print("=" * 60)

    results = db.search(cav_norm, k=args.k)

    # Display results
    print(f"\nTop {len(results)} results:")
    print("-" * 60)
    for i, (path, score) in enumerate(results, 1):
        filename = path.split('/')[-1]
        print(f"{i}. {filename:40s} -> {score:8.6f}")

    print("\n" + "=" * 60)
    print("Search completed!")
    print("=" * 60)

    # Optional: Save results to file
    if args.output:
        with open(args.output, 'w') as f:
            for path, score in results:
                f.write(f"{path}\t{score}\n")
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Image similarity search using CAV and FAISS"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Build database command
    build_parser = subparsers.add_parser('build-db', help='Build FAISS index from images')
    build_parser.add_argument(
        '--folder',
        required=True,
        help='Folder containing images to index'
    )
    build_parser.add_argument(
        '--output',
        default='image_db',
        help='Output prefix for index files (default: image_db)'
    )
    build_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    build_parser.set_defaults(func=cmd_build_db)

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar images')
    search_parser.add_argument(
        '--query',
        help='Folder containing query images (or combined pos+neg)'
    )
    search_parser.add_argument(
        '--query-positive',
        help='Folder containing positive example images'
    )
    search_parser.add_argument(
        '--query-negative',
        help='Folder containing negative example images'
    )
    search_parser.add_argument(
        '--db',
        required=True,
        help='Prefix of saved database (e.g., image_db)'
    )
    search_parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    search_parser.add_argument(
        '--output',
        help='Optional: Save results to file'
    )
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

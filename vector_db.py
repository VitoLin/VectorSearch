#!/usr/bin/env python3
"""
Simple Word2Vec + FAISS vector DB for word lookup.

Usage:
  Build index:
    python vector_db.py build --out-prefix word_index

  Query:
    python vector_db.py query --word king --out-prefix word_index --k 5

Dependencies:
  pip install gensim faiss-cpu numpy

This script:
  - trains a Word2Vec model on a tiny demo corpus (you can replace with your own)
  - builds a FAISS IndexFlatL2 index from word vectors
  - saves index, words list, and the gensim model
  - allows querying nearest words for a given word
"""
import os
import argparse
import pickle
from typing import List

import numpy as np
from gensim.models import Word2Vec
import faiss

# Small demo corpus. Replace or load your own corpus for better results.
DEMO_CORPUS = [
    "king queen man woman prince princess royal throne crown",
    "cat dog mouse pet animal feline canine",
    "apple orange banana fruit healthy sweet",
    "car truck vehicle drive road travel",
    "python java javascript programming language code developer",
    "music song melody rhythm instrument guitar piano",
    "sun moon star sky planet orbit space",
    "coffee tea drink beverage mug cup",
    "city town village urban rural street",
    "happy sad joy anger emotion feeling"
]


def train_word2vec(corpus: List[str], vector_size: int = 100, window: int = 5,
                   min_count: int = 1, epochs: int = 50, workers: int = 1) -> Word2Vec:
    sentences = [s.split() for s in corpus]
    model = Word2Vec(sentences=sentences, vector_size=vector_size,
                     window=window, min_count=min_count, workers=workers)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model


def build_faiss_index(model: Word2Vec, out_prefix: str):
    words = model.wv.index_to_key  # list of words in order
    d = model.wv.vector_size
    vectors = np.stack([model.wv.get_vector(w) for w in words]).astype('float32')

    index = faiss.IndexFlatL2(d)
    index.add(vectors)

    index_path = f"{out_prefix}.faiss"
    words_path = f"{out_prefix}_words.pkl"
    model_path = f"{out_prefix}.model"

    faiss.write_index(index, index_path)
    with open(words_path, "wb") as f:
        pickle.dump(words, f)
    model.save(model_path)

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved words list to {words_path}")
    print(f"Saved Word2Vec model to {model_path}")


def load_index_and_words(out_prefix: str):
    index_path = f"{out_prefix}.faiss"
    words_path = f"{out_prefix}_words.pkl"
    if not os.path.exists(index_path) or not os.path.exists(words_path):
        raise FileNotFoundError("Index or words file not found. Run build first.")
    index = faiss.read_index(index_path)
    with open(words_path, "rb") as f:
        words = pickle.load(f)
    return index, words


def query(word: str, out_prefix: str, k: int = 5):
    model_path = f"{out_prefix}.model"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Word2Vec model not found. Run build first.")
    model = Word2Vec.load(model_path)

    if word not in model.wv:
        print(f"Word '{word}' not in vocabulary.")
        return

    vec = model.wv.get_vector(word).astype('float32').reshape(1, -1)
    index, words = load_index_and_words(out_prefix)

    k = min(k, index.ntotal)
    distances, indices = index.search(vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((words[int(idx)], float(dist)))
    print(f"Nearest neighbors for '{word}':")
    for w, d in results:
        print(f"  {w}\t(distance={d:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Word2Vec + FAISS toy vector DB")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Train Word2Vec and build FAISS index")
    p_build.add_argument("--out-prefix", default="word_index", help="Output prefix for files")
    p_build.add_argument("--vector-size", type=int, default=100)
    p_build.add_argument("--epochs", type=int, default=50)

    p_query = sub.add_parser("query", help="Query nearest words")
    p_query.add_argument("--word", required=True, help="Word to query")
    p_query.add_argument("--out-prefix", default="word_index", help="Prefix used when building the index")
    p_query.add_argument("--k", type=int, default=5, help="Number of neighbors to return")

    args = parser.parse_args()

    if args.cmd == "build":
        print("Training Word2Vec model on demo corpus...")
        model = train_word2vec(DEMO_CORPUS, vector_size=args.vector_size, epochs=args.epochs)
        build_faiss_index(model, args.out_prefix)
    elif args.cmd == "query":
        query(args.word, args.out_prefix, k=args.k)


if __name__ == "__main__":
    main()

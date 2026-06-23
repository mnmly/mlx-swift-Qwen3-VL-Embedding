#!/usr/bin/env python3
"""Generate parity fixtures from the PyTorch/transformers reference.

Runs the reference ``Qwen3VLEmbedder`` and ``Qwen3VLReranker`` (from the
Qwen3-VL-Embedding repo) on a fixed set of inputs and dumps the resulting
embeddings + rerank scores as JSON. The Swift test suite loads these and checks
cosine >= 0.99 (embeddings) and |Δ| < 1e-2 (reranker scores).

Usage (from the python reference repo, which has torch/transformers installed):

    uv run python /path/to/gen_fixtures.py \
        --repo   /Users/mnmly/Development-local/GitHub/python/Qwen3-VL-Embedding \
        --embed-model    Qwen/Qwen3-VL-Embedding-2B \
        --rerank-model   Qwen/Qwen3-VL-Reranker-2B \
        --out    /path/to/mlx-swift-qwen3vl-embedding/Tests/Fixtures/parity.json

Model args accept a HF id or a local snapshot directory.
"""
import argparse
import json
import os
import sys


# Fixed inputs — keep in lockstep with the Swift parity test.
RERANK_QUERY = "What is the capital of France?"
RERANK_DOCUMENTS = [
    "Paris is the capital of France.",
    "The Great Wall of China is a famous landmark.",
    "Bananas are a good source of potassium.",
]
EMBED_TEXTS = [
    "A photograph of a golden retriever puppy.",
    "The Eiffel Tower at sunset.",
    "Quarterly revenue grew 12% year over year.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to the Qwen3-VL-Embedding repo (for src/).")
    ap.add_argument("--embed-model", required=True)
    ap.add_argument("--rerank-model", required=True)
    ap.add_argument("--image", default=None, help="Optional image path for a multimodal embedding.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "src"))
    from models.qwen3_vl_embedding import Qwen3VLEmbedder
    from models.qwen3_vl_reranker import Qwen3VLReranker

    fixtures: dict = {"inputs": {}, "reference": {}}

    # --- Reranker ---
    reranker = Qwen3VLReranker(args.rerank_model)
    scores = reranker.process(
        {
            "query": {"text": RERANK_QUERY},
            "documents": [{"text": d} for d in RERANK_DOCUMENTS],
        }
    )
    fixtures["inputs"]["rerank"] = {"query": RERANK_QUERY, "documents": RERANK_DOCUMENTS}
    fixtures["reference"]["rerank_scores"] = [float(s) for s in scores]
    del reranker

    # --- Embedder ---
    embedder = Qwen3VLEmbedder(args.embed_model)
    items = [{"text": t} for t in EMBED_TEXTS]
    inputs = {"embed_texts": EMBED_TEXTS}
    if args.image:
        items.append({"image": args.image})
        inputs["embed_image"] = args.image
    vectors = embedder.process(items, normalize=True)
    fixtures["inputs"]["embed"] = inputs
    fixtures["reference"]["embed_vectors"] = vectors.float().cpu().tolist()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fixtures, f, indent=2)
    print(f"wrote {args.out}: "
          f"{len(fixtures['reference']['rerank_scores'])} rerank scores, "
          f"{len(fixtures['reference']['embed_vectors'])} embed vectors")


if __name__ == "__main__":
    main()

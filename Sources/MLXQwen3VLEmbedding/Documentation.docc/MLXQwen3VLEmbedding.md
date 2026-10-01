# ``MLXQwen3VLEmbedding``

Multimodal embedding and reranking with Qwen3-VL on Apple Silicon, built on mlx-swift.

## Overview

This library wraps the Qwen3-VL backbone (from `mlx-swift-lm`) with the two thin heads from
the reference [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) repo:

- **Embedding** — last-token (EOS) pooling of the model's `last_hidden_state`, L2-normalized,
  with optional Matryoshka (MRL) truncation.
- **Reranking** — `sigmoid((W[yes] − W[no]) · h)` on the final prompt position's hidden state,
  in float32: the reference's binary linear head, i.e. `logits[yes] − logits[no]` without the
  bfloat16 rounding of the two logits.

The single entry point is ``Qwen3VLEmbeddingSession``: load a checkpoint once, then call
``Qwen3VLEmbeddingSession/embed(texts:instruction:)`` or ``Qwen3VLEmbeddingSession/rankedDocuments(query:documents:batchSize:)``.
The same session type drives both the `qwen3vl-embed` CLI and the SwiftUI demo app — all model
and prompt logic lives here, not in either frontend.

```swift
import MLXQwen3VLEmbedding

// Embedding (point at a Qwen3-VL-Embedding snapshot directory).
let embedder = try await Qwen3VLEmbeddingSession.load(
    .init(modelDirectory: embeddingModelURL, task: .embedding))
let vectors = try await embedder.embed(texts: ["a golden retriever puppy", "the Eiffel Tower"])

// Reranking (point at a Qwen3-VL-Reranker snapshot directory).
let reranker = try await Qwen3VLEmbeddingSession.load(
    .init(modelDirectory: rerankerModelURL, task: .reranker))
let ranked = try await reranker.rankedDocuments(
    query: .text("What is the capital of France?"),
    documents: [.text("Paris is the capital of France."), .text("Bananas are yellow.")])
```

Indexing a corpus of images goes through ``Qwen3VLEmbeddingSession/embed(images:instruction:batchSize:)``,
which folds several prompts into one language-model prefill:

```swift
let vectors = try await embedder.embed(images: chunk, batchSize: 16)
```

The vectors come back in input order and match the one-at-a-time path to within the
checkpoint's own bfloat16 rounding (in float32 the two agree to cosine 0.999996).

Concurrency: ``Qwen3VLEmbeddingSession`` is `Sendable` and backed by the actor
``Qwen3VLEmbeddingEngine``, so calls serialize safely; drive it from a single task for
predictable throughput.

## Topics

### Loading a model

- ``Qwen3VLEmbeddingSession``
- ``Qwen3VLEmbeddingSession/load(_:)``
- ``Qwen3VLEmbeddingConfig``
- ``Qwen3VLTask``

### Building inputs

- ``Qwen3VLContent``

### Embedding

- ``Qwen3VLEmbeddingSession/embed(texts:instruction:)``

### Indexing an image corpus

- ``Qwen3VLEmbeddingSession/embed(images:instruction:batchSize:)``
- ``Qwen3VLEmbeddingSession/embed(_:instruction:batchSize:)``

### Reranking

- ``Qwen3VLEmbeddingSession/rankedDocuments(query:documents:batchSize:)``

### Diagnostics and memory

- ``Qwen3VLEmbeddingSession/memorySnapshot()``
- ``Qwen3VLEmbeddingEngine/MemorySnapshot``

### Errors

- ``Qwen3VLEmbeddingError``

### Advanced

- ``Qwen3VLEmbeddingEngine``
- ``Qwen3VLModel``
- ``Qwen3VLBackbone``
```

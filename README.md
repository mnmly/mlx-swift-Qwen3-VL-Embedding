# mlx-swift-Qwen3-VL-Embedding

Apple-Silicon port of [QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
& Qwen3-VL-Reranker, built on [mlx-swift](https://github.com/ml-explore/mlx-swift) and
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

📖 **[API documentation](https://mnmly.github.io/mlx-swift-Qwen3-VL-Embedding/)**

![Qwen3-VL Embedding & Reranker demo — cross-modal (text + image) embedding on the left, reranking on the right](assets/demo.png)

The Qwen3-VL **backbone** (vision tower + language model, DeepStack, interleaved-MRoPE, QK-norm)
already ships in `mlx-swift-lm` as `MLXVLM.Qwen3VL`. This package adds the two thin heads the
reference repo layers on top:

| Head | What it does | Needs vendored backbone? |
|---|---|---|
| **Embedder** | last-token (EOS) pooling of `last_hidden_state` → L2-normalize → optional Matryoshka (MRL) truncation | Yes — the public `Qwen3VL` exposes only logits, not the pre-`lm_head` hidden state |
| **Reranker** | `sigmoid((W[yes] − W[no]) · h)` on the last position's `last_hidden_state`, in float32 (the reference's binary linear head; equal to `logits[yes] − logits[no]` without bfloat16 logit rounding). Text-only pairs are scored in right-padded micro-batches | Yes — same reason; reading the two logits from the bfloat16 vocabulary projection put every score on a 1/16 logit grid |

A single library-side `Qwen3VLEmbeddingSession` drives both the `qwen3vl-embed` CLI and the SwiftUI
demo app (the *shared-driver* pattern). Text, image, and cross-modal embeddings are supported.

## Status

Text embeddings, image embeddings, and reranking are **numerically validated** against the
PyTorch/transformers reference (enforced by `ParityTests`):

| Path | Parity vs Python reference |
|---|---|
| Reranker score | \|Δ\| < 0.008 on the fixture; median 0.006 / max 0.036 over 93 queries × 50 passages |
| Text embedding (cosine) | ≥ 0.9997 |
| Image embedding (cosine) | ≥ 0.997 |

See `Sources/MLXQwen3VLEmbedding/Vendored/VENDORED.md` for the vendored backbone provenance and the
local fixes (notably a `gelu_pytorch_tanh` vision-MLP fix that should be reported upstream).

## Usage (`Qwen3VLEmbeddingSession`)

```swift
import MLXQwen3VLEmbedding

let embedder = try await Qwen3VLEmbeddingSession.load(
    .init(modelDirectory: embeddingModelURL, task: .embedding))
let textVecs = try await embedder.embed(texts: ["a golden retriever puppy", "the Eiffel Tower"])
let imageVec = try await embedder.embed(.image(cgImage))            // same space as text

// Indexing a corpus: one language-model prefill per micro-batch, vectors in input order.
let vecs = try await embedder.embed(images: chunkOfCGImages, batchSize: 16)

let reranker = try await Qwen3VLEmbeddingSession.load(
    .init(modelDirectory: rerankerModelURL, task: .reranker))
let ranked = try await reranker.rankedDocuments(
    query: .text("capital of France"),
    documents: [.text("Paris is the capital of France."), .text("Bananas are yellow.")])
```

## Indexing an image corpus

`embed(images:batchSize:)` folds several prompts into one language-model prefill. Measured on an
M5 Max (128 GB), embedding 128 images end-to-end, best of three passes:

| Corpus | patches/image | one at a time | batch 8 | batch 32 |
|---|---|---|---|---|
| 512×384, uniform | 768 | 14.6 img/s | 25.6 | **25.8** (1.77×) |
| BL Books originals, mixed | 190–2 900 | 11.8 img/s | 13.8 | **14.4** (1.22×) |
| ≤1400 px edge, mixed | ~1 000–5 100 | 4.44 img/s | 4.48 | **4.51** (1.02×) |
| 1280×1024 (the processor's pixel cap) | 5 120 | 3.17 img/s | **3.28** (1.03×) | — |

The pattern is the point: batching pays when a single image is too small to fill the GPU, and a
native-resolution image large enough to hit the processor's pixel cap already fills it on its own.
Peak GPU stays between 4.6 GB and 6.6 GB across the whole sweep.

The *vision tower* is deliberately **not** batched — see the header of
`Sources/MLXQwen3VLEmbedding/Qwen3VLBatchedForward.swift` for the measurements behind that.

Batched vectors match the one-at-a-time path to cosine **0.999996** in float32
(`Qwen3VLEmbeddingConfig.computeDType = .float32`). In the shipped bfloat16 they agree to 0.9989,
which is well inside the checkpoint's own quantization noise: against a float32 reference, the
unbatched path scores 0.9987 and the batched path 0.9987 — batching costs no accuracy.
`BatchParityTests` enforces both bars.

## CLI

```sh
qwen3vl-embed embed  --model <Embedding-2B-dir> "a dog" "a cat" --image photo.jpg
qwen3vl-embed rerank --model <Reranker-2B-dir>  "capital of France" "Paris is the capital." "…"
qwen3vl-embed bench  --model <Embedding-2B-dir> --iterations 20   # throughput + leak watch
qwen3vl-embed image-bench --model <Embedding-2B-dir> --images <dir> \
  --count 128 --batch-sizes 1,4,8,16,32                           # batch sweep + parity
qwen3vl-embed rerank-bench --model <Reranker-2B-dir> --input items.json \
  --batch-sizes 1,4,8,16 --scores scores.json   # rerank latency sweep + batch parity
```

## Build & test

```sh
xcodebuild -scheme mlx-swift-qwen3vl-embedding-Package -destination 'platform=macOS' \
  -derivedDataPath .xcdd build
xcodebuild -scheme mlx-swift-qwen3vl-embedding-Package -destination 'platform=macOS' test
```

Use `xcodebuild` (not `swift test`, which can't load MLX's Metal library). The parity tests
auto-discover models from the Hugging Face cache and skip cleanly when absent.

## Documentation

📖 **[API reference (DocC)](https://mnmly.github.io/mlx-swift-Qwen3-VL-Embedding/)** — built and
deployed from `main` by `.github/workflows/docs.yml`. Build locally with `Scripts/build_docs.sh`
(`Scripts/build_docs.sh preview` for live reload).

## Weights

Point the session at a local snapshot of `Qwen/Qwen3-VL-Embedding-2B` (or `-Reranker-2B`, or the
8B variants). bf16 HuggingFace safetensors load directly — no separate conversion step. The
embedding model's chat template appends `<|endoftext|>` as the pooled EOS token (handled
automatically).

## License

[MIT](LICENSE) © 2026 Hiroaki Yamane.

This package vendors a modified copy of the Qwen3-VL model from
[`ml-explore/mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) (MIT) and is a port of the
[Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) reference (Apache-2.0). See
[`NOTICE`](NOTICE) for attributions and `Sources/MLXQwen3VLEmbedding/Vendored/VENDORED.md` for the
vendored-backbone provenance and modifications. Model weights are distributed separately by their
authors under their own licenses.

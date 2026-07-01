// swift-tools-version: 6.1
//
// mlx-swift-qwen3vl-embedding — Apple-Silicon port of QwenLM/Qwen3-VL-Embedding
// & Qwen3-VL-Reranker on top of mlx-swift / mlx-swift-lm.
//
// The Qwen3-VL *backbone* (vision tower + language model) already ships in
// mlx-swift-lm (`MLXVLM.Qwen3VL`). This package adds the two thin heads the
// reference repo layers on top:
//
//   • Embedder — last-token (EOS) pooling of `last_hidden_state`, L2-normalize,
//     optional Matryoshka (MRL) truncation. Needs the pre-`lm_head` hidden state,
//     which the public `Qwen3VL` does not expose, so a copy of the backbone is
//     vendored under Sources/MLXQwen3VLEmbedding/Vendored (see VENDORED.md).
//   • Reranker — sigmoid(logits[yes] − logits[no]) at the last position. Uses the
//     stock `MLXVLM.Qwen3VL` logits directly; no vendored code required.
//
// A single library-side `Qwen3VLEmbeddingSession` drives both the CLI and the
// SwiftUI app (the swift-cli-gui-shared-driver pattern).

import PackageDescription

let package = Package(
    name: "mlx-swift-qwen3vl-embedding",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "MLXQwen3VLEmbedding", targets: ["MLXQwen3VLEmbedding"]),
        .executable(name: "qwen3vl-embed", targets: ["qwen3vl-embed"]),
    ],
    dependencies: [
        // Remote, versioned dependencies. Both reference the same
        // github.com/ml-explore/mlx-swift, so there is a single `mlx-swift` identity in the
        // graph (no local-path override conflict).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.4"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "MLXQwen3VLEmbedding",
            dependencies: [
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/MLXQwen3VLEmbedding",
            exclude: ["Vendored/VENDORED.md"]
        ),
        .executableTarget(
            name: "qwen3vl-embed",
            dependencies: [
                "MLXQwen3VLEmbedding",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Examples/qwen3vl-embed"
        ),
        // The SwiftUI macOS demo lives in Examples/Qwen3VLEmbeddingDemo/Qwen3VLEmbeddingDemo.xcodeproj
        // (a real app target referencing the MLXQwen3VLEmbedding library product). Both it and the
        // qwen3vl-embed CLI drive the same library-side ``Qwen3VLEmbeddingSession``.
        .testTarget(
            name: "MLXQwen3VLEmbeddingTests",
            dependencies: ["MLXQwen3VLEmbedding"],
            path: "Tests/MLXQwen3VLEmbeddingTests"
        ),
    ]
)

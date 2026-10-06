// Copyright © 2026
//
// Umbrella + scaffolding for MLXQwen3VLEmbedding. Real surface (Session, heads,
// vendored backbone) lands in the following phases; this file pins the package
// identity and force-links the upstream products the package builds on.

import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXVLM

public enum MLXQwen3VLEmbedding {
    /// Package version, bumped manually alongside releases.
    public static let version = "0.3.1"
}

/// Which Qwen3-VL head the session should run.
public enum Qwen3VLTask: String, Sendable, CaseIterable {
    /// Multimodal embedding (last-token pooling + L2 + optional MRL truncation).
    case embedding
    /// Query/document relevance scoring: sigmoid of `(W[yes] − W[no]) · h` in float32, the
    /// reference's binary head (`logits[yes] − logits[no]` without bfloat16 logit rounding).
    case reranker
}

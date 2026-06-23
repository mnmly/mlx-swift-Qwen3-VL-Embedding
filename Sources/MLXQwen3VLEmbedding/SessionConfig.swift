// Copyright © 2026
//
// Configuration for a ``Qwen3VLEmbeddingSession``. Every knob that affects loading,
// preprocessing, or the head math lives here with defaults that work for both the
// CLI and a SwiftUI frontend (the swift-cli-gui-shared-driver pattern).

import Foundation

public struct Qwen3VLEmbeddingConfig: Sendable {
    /// Directory holding a model snapshot (config.json + *.safetensors + tokenizer + preprocessor_config.json).
    public var modelDirectory: URL

    /// Which head to run.
    public var task: Qwen3VLTask

    /// Token-budget bounds for image resizing (`min_pixels` / `max_pixels` in the reference).
    public var minPixels: Int
    public var maxPixels: Int

    /// Truncation cap on the assembled prompt (reference: 8192 embed, 10240 rerank).
    public var maxLength: Int

    /// L2-normalize the embedding (embedder only).
    public var normalize: Bool

    /// Matryoshka (MRL) truncation dimension for the embedding, or `nil` for the full vector.
    public var embeddingDimension: Int?

    /// Instruction injected into the system prompt; `nil` uses the task default.
    public var instruction: String?

    /// Bound on MLX's reusable Metal buffer cache (bytes); `nil` leaves the MLX default.
    public var gpuCacheLimit: Int?

    public init(
        modelDirectory: URL,
        task: Qwen3VLTask,
        // Reference token budgets with IMAGE_FACTOR = 32: 4·32² and 1800·32².
        minPixels: Int = 4 * 32 * 32,
        maxPixels: Int = 1800 * 32 * 32,
        maxLength: Int? = nil,
        normalize: Bool = true,
        embeddingDimension: Int? = nil,
        instruction: String? = nil,
        gpuCacheLimit: Int? = 512 * 1024 * 1024
    ) {
        self.modelDirectory = modelDirectory
        self.task = task
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.maxLength = maxLength ?? (task == .reranker ? 10240 : 8192)
        self.normalize = normalize
        self.embeddingDimension = embeddingDimension
        self.instruction = instruction
        self.gpuCacheLimit = gpuCacheLimit
    }

    /// The instruction to use, falling back to the task default.
    public var resolvedInstruction: String {
        if let instruction, !instruction.isEmpty { return instruction }
        return task.defaultInstruction
    }
}

extension Qwen3VLTask {
    /// Default instruction strings from the reference repo.
    public var defaultInstruction: String {
        switch self {
        case .embedding:
            return "Represent the user's input."
        case .reranker:
            return "Given a search query, retrieve relevant candidates that answer the query."
        }
    }
}

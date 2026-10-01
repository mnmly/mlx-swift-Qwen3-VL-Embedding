// Copyright © 2026
//
// The single library-side driver consumed identically by the CLI and a SwiftUI
// app (the swift-cli-gui-shared-driver pattern). It loads the model once and
// exposes the embedding / reranking heads; no model/engine setup lives in either
// frontend. Cadence, autoreleasepool, and MainActor hops stay frontend-side.

import CoreGraphics
import Foundation

/// A ready-to-go Qwen3-VL embedding / reranking session.
///
/// Contract: one stepping caller at a time. The underlying ``Qwen3VLEmbeddingEngine``
/// is an actor, so concurrent calls serialize safely; throughput-sensitive callers
/// should still drive it from a single task.
public struct Qwen3VLEmbeddingSession: Sendable {
    public let config: Qwen3VLEmbeddingConfig
    private let engine: Qwen3VLEmbeddingEngine

    private init(config: Qwen3VLEmbeddingConfig, engine: Qwen3VLEmbeddingEngine) {
        self.config = config
        self.engine = engine
    }

    /// Load the model and return a ready-to-use session.
    public static func load(_ config: Qwen3VLEmbeddingConfig) async throws -> Qwen3VLEmbeddingSession {
        let engine = try await Qwen3VLEmbeddingEngine(config: config)
        return Qwen3VLEmbeddingSession(config: config, engine: engine)
    }

    // MARK: - Embedding

    /// Embed each content item into a (optionally L2-normalized, optionally MRL-truncated)
    /// vector. Requires a session loaded with `task: .embedding`.
    ///
    /// `instruction`, when non-nil, overrides the session's configured/default
    /// instruction for this call only — pass a retrieval task instruction on the
    /// query side to do asymmetric query↔document retrieval on one loaded session.
    /// `nil` (the default) keeps the prior session-wide behaviour.
    public func embed(
        _ contents: [Qwen3VLContent], instruction: String? = nil
    ) async throws -> [[Float]] {
        try await engine.embed(contents, instruction: instruction)
    }

    /// Text-only convenience.
    public func embed(texts: [String], instruction: String? = nil) async throws -> [[Float]] {
        try await embed(texts.map { .text($0) }, instruction: instruction)
    }

    /// Embed a single content item.
    public func embed(_ content: Qwen3VLContent, instruction: String? = nil) async throws -> [Float] {
        try await embed([content], instruction: instruction)[0]
    }

    /// Embed many items with `batchSize` of them per model call — the corpus-indexing
    /// entry point.
    ///
    /// Identical math to the unbatched `embed(_:instruction:)` — prompts are right-padded to
    /// the batch's longest and pooled at each row's own final token, so padding never reaches
    /// a vector — but with one language-model prefill per micro-batch instead of one per
    /// item. Vectors come back in input order.
    ///
    /// - Parameters:
    ///   - contents: the items to embed.
    ///   - instruction: per-call instruction override, as in the unbatched overload.
    ///   - batchSize: items per model call; `1` reproduces the unbatched path. 8–16 is a
    ///     good starting point for a mixed-resolution image corpus.
    public func embed(
        _ contents: [Qwen3VLContent], instruction: String? = nil, batchSize: Int
    ) async throws -> [[Float]] {
        try await engine.embed(contents, instruction: instruction, batchSize: batchSize)
    }

    /// Image-only convenience over ``embed(_:instruction:batchSize:)``.
    ///
    /// - Parameters:
    ///   - images: images to embed, one vector each, returned in input order.
    ///   - instruction: per-call instruction override.
    ///   - batchSize: images per model call.
    public func embed(
        images: [CGImage], instruction: String? = nil, batchSize: Int = 8
    ) async throws -> [[Float]] {
        try await embed(images.map { .image($0) }, instruction: instruction, batchSize: batchSize)
    }

    // MARK: - Reranking

    /// Relevance scores in document order for one query against many documents.
    ///
    /// - Parameters:
    ///   - query: the query, scored against every document.
    ///   - documents: the candidates.
    ///   - batchSize: text-only pairs per model call (pairs with images run one at a time);
    ///     `1` scores one pair per call. The default was the fastest measured for ~1,000-character
    ///     passages; batched and unbatched scores agree.
    public func rerank(
        query: Qwen3VLContent, documents: [Qwen3VLContent], batchSize: Int = 8
    ) async throws -> [Float] {
        try await engine.rerank(query: query, documents: documents, batchSize: batchSize)
    }

    /// Text-only convenience: score `query` against each document string.
    ///
    /// - Parameters:
    ///   - query: the query text.
    ///   - documents: the candidate texts.
    ///   - batchSize: pairs per model call, as in ``rerank(query:documents:batchSize:)-(Qwen3VLContent,_,_)``.
    public func rerank(query: String, documents: [String], batchSize: Int = 8) async throws -> [Float] {
        try await rerank(
            query: .text(query), documents: documents.map { .text($0) }, batchSize: batchSize)
    }

    /// Documents ranked best-first as `(originalIndex, score)`.
    ///
    /// - Parameters:
    ///   - query: the query, scored against every document.
    ///   - documents: the candidates.
    ///   - batchSize: text-only pairs per model call, as in ``rerank(query:documents:batchSize:)-(Qwen3VLContent,_,_)``.
    public func rankedDocuments(
        query: Qwen3VLContent, documents: [Qwen3VLContent], batchSize: Int = 8
    ) async throws -> [(index: Int, score: Float)] {
        let scores = try await rerank(query: query, documents: documents, batchSize: batchSize)
        return scores.enumerated()
            .map { (index: $0.offset, score: $0.element) }
            .sorted { $0.score > $1.score }
    }

    // MARK: - Memory

    /// Current MLX GPU memory snapshot (flat `active` across runs ⇒ no leak).
    public static func memorySnapshot() -> Qwen3VLEmbeddingEngine.MemorySnapshot {
        Qwen3VLEmbeddingEngine.memorySnapshot()
    }
}

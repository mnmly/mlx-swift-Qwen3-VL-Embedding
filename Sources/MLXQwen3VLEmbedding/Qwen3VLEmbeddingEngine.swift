// Copyright © 2026
//
// Loads a Qwen3-VL checkpoint once and runs the embedding / reranking heads.
// Thin bridge over mlx-swift-lm; all prompt assembly lives in
// ``Qwen3VLPromptBuilder`` and the head math here.

import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

public enum Qwen3VLEmbeddingError: Error, CustomStringConvertible {
    case emptyDocuments
    case tokenNotFound(String)
    case prepareProducedTokens
    case embedderNotLoaded

    public var description: String {
        switch self {
        case .emptyDocuments: return "rerank requires at least one document"
        case .tokenNotFound(let t): return "could not resolve a single token id for \"\(t)\""
        case .prepareProducedTokens: return "model.prepare returned tokens, expected logits"
        case .embedderNotLoaded:
            return "this engine was loaded for reranking; load with task .embedding to embed"
        }
    }
}

/// A loaded Qwen3-VL engine. One stepping caller at a time (actor-isolated).
public actor Qwen3VLEmbeddingEngine {
    public let container: ModelContainer
    public let config: Qwen3VLEmbeddingConfig
    /// Resolved snapshot directory (used to read the reranker's `1_LogitScore` head config).
    public let modelDirectory: URL

    /// Load the model once and reuse it across calls. `gpuCacheLimit` bounds MLX's
    /// reusable Metal buffer cache so a long-lived process doesn't keep the multi-GB
    /// peak footprint resident (the cache is not a leak, but it is large).
    public init(config: Qwen3VLEmbeddingConfig) async throws {
        if let limit = config.gpuCacheLimit {
            MLX.Memory.cacheLimit = limit
        }
        self.config = config
        let directory = Qwen3VLModel.resolveSnapshotDirectory(config.modelDirectory)
        self.modelDirectory = directory
        let loader = #huggingFaceTokenizerLoader()
        switch config.task {
        case .embedding:
            // Vendored backbone exposes `lastHiddenState` for last-token pooling. The wrapped
            // loader suppresses the tokenizer's auto-appended `<|endoftext|>` on `encode(text:)`,
            // which otherwise breaks stock image-placeholder matching (see NoAutoSpecialsTokenizer).
            self.container = try await Qwen3VLEmbeddingFactory.shared.loadContainer(
                from: directory, using: Qwen3VLEmbeddingTokenizerLoader(base: loader))
        case .reranker:
            // Stock `Qwen3VL` logits are sufficient for the yes/no score.
            self.container = try await VLMModelFactory.shared.loadContainer(
                from: directory, using: loader)
        }
    }

    // MARK: - Embedding

    /// Embed each content item: run the backbone, take the last-token (EOS) hidden
    /// state, optionally truncate to the Matryoshka dimension, optionally L2-normalize.
    /// Mirrors the reference `Qwen3VLEmbedder._pooling_last` + `F.normalize`.
    ///
    /// `instruction`, when non-nil, overrides `config.resolvedInstruction` for THIS
    /// call only — the lever for asymmetric retrieval: embed a search *query* with a
    /// task instruction ("retrieve passages that answer this") distinct from how the
    /// *documents* were embedded, without loading a second session. `nil` preserves
    /// the prior behaviour (the session-wide configured/default instruction).
    public func embed(
        _ contents: [Qwen3VLContent], instruction: String? = nil
    ) async throws -> [[Float]] {
        let instruction = instruction ?? config.resolvedInstruction
        let minPixels = config.minPixels
        let maxPixels = config.maxPixels
        let normalize = config.normalize
        let dimension = config.embeddingDimension

        return try await container.perform { (context: ModelContext) in
            guard let backbone = context.model as? Qwen3VLBackbone else {
                throw Qwen3VLEmbeddingError.embedderNotLoaded
            }
            // The reference tokenizes the chat-templated text with add_special_tokens=True,
            // appending `<|endoftext|>` as the EOS that last-token pooling reads. Swift's
            // applyChatTemplate stops at "assistant\n", so append it explicitly.
            let eosId = context.tokenizer.convertTokenToId("<|endoftext|>") ?? 151_643

            var vectors: [[Float]] = []
            vectors.reserveCapacity(contents.count)
            for content in contents {
                let input = Qwen3VLPromptBuilder.embeddingUserInput(
                    content: content, instruction: instruction,
                    minPixels: minPixels, maxPixels: maxPixels)
                let prepared = try await context.processor.prepare(input: input)
                let lmInput = Self.appendingEmbeddingEOS(prepared, eosId: eosId)
                let cache = backbone.newCache(parameters: nil)
                // [1, seq, hidden]
                let hidden = try backbone.lastHiddenState(lmInput, cache: cache)
                let seq = hidden.dim(1)
                var pooled = hidden[0, seq - 1].asType(.float32)  // last token, [hidden]
                if let dimension { pooled = pooled[0 ..< dimension] }
                if normalize {
                    let norm = sqrt((pooled * pooled).sum())
                    pooled = pooled / maximum(norm, MLXArray(Float(1e-12)))
                }
                MLX.eval(pooled)
                vectors.append(pooled.asArray(Float.self))
            }
            return vectors
        }
    }

    /// Append the embedding EOS (`<|endoftext|>`) the reference pools, unless the prompt
    /// already ends with it. Preserves any image/video/audio payload.
    private static func appendingEmbeddingEOS(_ input: LMInput, eosId: Int) -> LMInput {
        let tokens = input.text.tokens  // [1, seq]
        let seq = tokens.dim(1)
        if tokens[0, seq - 1].item(Int.self) == eosId { return input }
        let eos = MLXArray([Int32(eosId)]).reshaped([1, 1]).asType(tokens.dtype)
        let extended = concatenated([tokens, eos], axis: 1)
        let mask = ones(like: extended).asType(.int8)
        return LMInput(
            text: .init(tokens: extended, mask: mask),
            image: input.image, video: input.video)
    }

    // MARK: - Reranking

    /// Score each `(query, document)` pair: `sigmoid(logits[yes] − logits[no])` at the
    /// final prompt position. Uses the stock `Qwen3VL` logits — no vendored backbone.
    public func rerank(
        query: Qwen3VLContent, documents: [Qwen3VLContent]
    ) async throws -> [Float] {
        guard !documents.isEmpty else { throw Qwen3VLEmbeddingError.emptyDocuments }

        let instruction = config.resolvedInstruction
        let minPixels = config.minPixels
        let maxPixels = config.maxPixels
        let directory = modelDirectory

        return try await container.perform { (context: ModelContext) in
            let (yesId, noId) = try Self.rerankerTokenIds(
                directory: directory, tokenizer: context.tokenizer)

            var scores: [Float] = []
            scores.reserveCapacity(documents.count)
            for document in documents {
                let input = Qwen3VLPromptBuilder.rerankUserInput(
                    query: query, document: document, instruction: instruction,
                    minPixels: minPixels, maxPixels: maxPixels)
                let lmInput = try await context.processor.prepare(input: input)
                let cache = context.model.newCache(parameters: nil)
                guard
                    case .logits(let out) = try context.model.prepare(
                        lmInput, cache: cache, windowSize: nil)
                else { throw Qwen3VLEmbeddingError.prepareProducedTokens }

                let length = lmInput.text.tokens.dim(-1)
                let diff = (out.logits[0, length - 1, yesId] - out.logits[0, length - 1, noId])
                    .asType(.float32)
                MLX.eval(diff)
                let d = Double(diff.item(Float.self))
                scores.append(Float(1.0 / (1.0 + exp(-d))))
            }
            return scores
        }
    }

    private struct LogitScoreConfig: Decodable {
        let trueTokenId: Int
        let falseTokenId: Int
        enum CodingKeys: String, CodingKey {
            case trueTokenId = "true_token_id"
            case falseTokenId = "false_token_id"
        }
    }

    /// The (yes, no) token ids for the reranker head. Prefer the authoritative
    /// `1_LogitScore/config.json` shipped with the checkpoint; fall back to the
    /// tokenizer's `"yes"`/`"no"` lookup (matching the reference `get_vocab()[…]`).
    private static func rerankerTokenIds(
        directory: URL, tokenizer: any MLXLMCommon.Tokenizer
    ) throws -> (yes: Int, no: Int) {
        let url = directory.appending(components: "1_LogitScore", "config.json")
        if let data = try? Data(contentsOf: url),
            let cfg = try? JSONDecoder().decode(LogitScoreConfig.self, from: data)
        {
            return (cfg.trueTokenId, cfg.falseTokenId)
        }
        return (try tokenId(for: "yes", in: tokenizer), try tokenId(for: "no", in: tokenizer))
    }

    /// Resolve a vocabulary token id, matching the Python `tokenizer.get_vocab()[token]`
    /// single-token lookup.
    private static func tokenId(for token: String, in tokenizer: any MLXLMCommon.Tokenizer) throws -> Int {
        if let id = tokenizer.convertTokenToId(token) { return id }
        let encoded = tokenizer.encode(text: token, addSpecialTokens: false)
        guard encoded.count == 1 else { throw Qwen3VLEmbeddingError.tokenNotFound(token) }
        return encoded[0]
    }

    // MARK: - Memory

    public struct MemorySnapshot: Sendable {
        public let active: Int
        public let cache: Int
        public let peak: Int
    }

    public static func memorySnapshot() -> MemorySnapshot {
        MemorySnapshot(
            active: MLX.Memory.activeMemory, cache: MLX.Memory.cacheMemory,
            peak: MLX.Memory.peakMemory)
    }
}

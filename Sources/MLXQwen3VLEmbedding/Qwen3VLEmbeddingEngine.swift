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
import MLXNN
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
            if let computeDType = config.computeDType {
                // Cast the checkpoint's floating-point parameters (see
                // ``Qwen3VLEmbeddingConfig/computeDType``).
                await self.container.perform { (ctx: ModelContext) in
                    _ = (ctx.model as? Qwen3VLBackbone)?.apply {
                        $0.dtype.isFloatingPoint ? $0.asType(computeDType) : $0
                    }
                }
            }
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

    // MARK: - Batched embedding

    /// Embed many items with `batchSize` of them in each model call.
    ///
    /// Same math as the unbatched `embed(_:instruction:)`, but with one language-model
    /// prefill per micro-batch instead of one per item — what makes corpus-scale image
    /// indexing practical. Prompts are right-padded to the batch's longest and pooled at
    /// each row's own final token, so the padding never reaches a vector. Results come back
    /// in input order.
    ///
    /// Items are grouped by image area so a micro-batch holds similarly sized images and the
    /// right-padding stays short. Items carrying more than one image fall back to the
    /// unbatched path: the batch assembly assumes one grid per row, which is what lets the
    /// image features line up with the image-token slots by a single running count.
    ///
    /// Preprocessing happens one micro-batch at a time, so peak memory tracks `batchSize`,
    /// not `contents.count` — but the returned vectors do accumulate, so prefer feeding a
    /// long corpus in windows of a few hundred.
    ///
    /// - Parameters:
    ///   - contents: the items to embed.
    ///   - instruction: per-call instruction override, as in the unbatched overload.
    ///   - batchSize: items per model call. 1 reproduces the unbatched path.
    public func embed(
        _ contents: [Qwen3VLContent], instruction: String? = nil, batchSize: Int
    ) async throws -> [[Float]] {
        guard batchSize > 1, contents.count > 1 else {
            return try await embed(contents, instruction: instruction)
        }

        let instruction = instruction ?? config.resolvedInstruction
        let minPixels = config.minPixels
        let maxPixels = config.maxPixels
        let normalize = config.normalize
        let dimension = config.embeddingDimension
        let patchBudget = config.maxBatchPatches

        // Batchable items carry at most one image; the rest go through the single path.
        let batchable = contents.indices.filter { contents[$0].images.count <= 1 }
        let singles = contents.indices.filter { contents[$0].images.count > 1 }

        // Group by image area: a micro-batch of similarly sized images pads little. Native
        // resolution means area is only a proxy for token count, but a close one — the
        // processor rounds each side to a multiple of patch × merge.
        let ordered = batchable.sorted {
            let (a, b) = (Self.imageArea(contents[$0]), Self.imageArea(contents[$1]))
            return a == b ? $0 < $1 : a < b
        }

        var results = [[Float]?](repeating: nil, count: contents.count)

        let batched: [(Int, [Float])] = try await container.perform {
            (context: ModelContext) in
            guard let backbone = context.model as? Qwen3VLBackbone else {
                throw Qwen3VLEmbeddingError.embedderNotLoaded
            }
            let eosId = context.tokenizer.convertTokenToId("<|endoftext|>") ?? 151_643
            // The M-RoPE position ids depend only on the prompt tokens and the image grid,
            // so a corpus of similarly shaped images pays `getRopeIndex`'s host-side scan
            // once per shape rather than once per image. The grid has to be part of the key:
            // two different grids can carry the same token count (22×40 and 20×44 are both
            // 880 patches) yet give the visual tokens different h/w positions.
            var positionCache: [PositionKey: MLXArray] = [:]
            var out: [(Int, [Float])] = []
            out.reserveCapacity(ordered.count)

            var cursor = 0
            while cursor < ordered.count {
                let window = Array(ordered[cursor ..< min(cursor + batchSize, ordered.count)])
                cursor += window.count

                var rows: [PreparedRow] = []
                rows.reserveCapacity(window.count)
                for index in window {
                    let input = Qwen3VLPromptBuilder.embeddingUserInput(
                        content: contents[index], instruction: instruction,
                        minPixels: minPixels, maxPixels: maxPixels)
                    let prepared = try await context.processor.prepare(input: input)
                    let lmInput = Self.appendingEmbeddingEOS(prepared, eosId: eosId)
                    rows.append(
                        PreparedRow(
                            index: index,
                            tokens: lmInput.text.tokens.asArray(Int32.self),
                            pixels: lmInput.image?.pixels,
                            grid: lmInput.image?.frames?.first))
                }

                for slice in Self.splitToPatchBudget(rows, budget: patchBudget) {
                    out.append(
                        contentsOf: try Self.embedOneBatch(
                            slice, backbone: backbone, eosId: eosId,
                            positionCache: &positionCache,
                            normalize: normalize, dimension: dimension))
                }
            }
            return out
        }

        for (index, vector) in batched { results[index] = vector }
        for index in singles {
            results[index] = try await embed([contents[index]], instruction: instruction)[0]
        }
        return results.map { $0 ?? [] }
    }

    /// Cache key for a prompt's M-RoPE position ids: the tokens *and* the image grid that
    /// produced them.
    private struct PositionKey: Hashable {
        let tokens: [Int32]
        let grid: [Int]

        init(tokens: [Int32], grid: THW?) {
            self.tokens = tokens
            self.grid = grid.map { [$0.t, $0.h, $0.w] } ?? []
        }
    }

    /// One prepared prompt: host-side tokens (so padding is a plain array splice) plus the
    /// image payload the processor produced.
    private struct PreparedRow {
        let index: Int
        let tokens: [Int32]
        let pixels: MLXArray?
        let grid: THW?
    }

    /// Assemble one right-padded batch, run it, and pool per row.
    private static func embedOneBatch(
        _ rows: [PreparedRow],
        backbone: Qwen3VLBackbone,
        eosId: Int,
        positionCache: inout [PositionKey: MLXArray],
        normalize: Bool,
        dimension: Int?
    ) throws -> [(Int, [Float])] {
        let batch = rows.count
        let sequence = rows.map(\.tokens.count).max() ?? 0

        // Right-pad the prompts. Attention is causal, so the filler never reaches a real
        // token; it must merely not be an image token (pooling reads `length - 1`).
        var flatTokens: [Int32] = []
        flatTokens.reserveCapacity(batch * sequence)
        var positionRows: [MLXArray] = []
        positionRows.reserveCapacity(batch)

        for row in rows {
            flatTokens.append(contentsOf: row.tokens)
            flatTokens.append(
                contentsOf: repeatElement(Int32(eosId), count: sequence - row.tokens.count))

            let key = PositionKey(tokens: row.tokens, grid: row.grid)
            let cached = positionCache[key]
            let positions =
                cached
                ?? Qwen3VLLanguage.getRopeIndex(
                    inputIds: MLXArray(row.tokens, [1, row.tokens.count]),
                    imageGridTHW: row.grid.map { [$0] },
                    videoGridTHW: nil,
                    spatialMergeSize: backbone.config.visionConfiguration.spatialMergeSize,
                    imageTokenId: backbone.config.imageTokenIndex,
                    videoTokenId: backbone.config.videoTokenIndex,
                    visionStartTokenId: backbone.config.visionStartTokenId,
                    attentionMask: nil
                ).0
            if cached == nil { positionCache[key] = positions }

            positionRows.append(padColumns(positions, to: sequence))
        }

        // The vision tower runs one image at a time, so the patch rows go in unpadded.
        let visionRows = rows.compactMap { row -> Qwen3VLBackbone.VisionRow? in
            guard let pixels = row.pixels, let grid = row.grid else { return nil }
            return .init(grid: grid, pixels: pixels)
        }

        let hidden = backbone.lastHiddenStateBatch(
            .init(
                tokens: MLXArray(flatTokens, [batch, sequence]),
                positionIds: concatenated(positionRows, axis: 1),
                visionRows: visionRows))

        // Pool each row at its own final token rather than the padded last column.
        let last = MLXArray(rows.map { Int32($0.tokens.count - 1) }).reshaped([batch, 1, 1])
        var pooled = takeAlong(hidden, broadcast(last, to: [batch, 1, hidden.dim(2)]), axis: 1)
            .squeezed(axis: 1)
            .asType(.float32)
        if let dimension { pooled = pooled[0..., 0 ..< dimension] }
        if normalize {
            let norm = sqrt((pooled * pooled).sum(axis: 1, keepDims: true))
            pooled = pooled / maximum(norm, MLXArray(Float(1e-12)))
        }
        MLX.eval(pooled)

        let width = pooled.dim(1)
        let flat = pooled.asArray(Float.self)
        return rows.enumerated().map { offset, row in
            (row.index, Array(flat[(offset * width) ..< ((offset + 1) * width)]))
        }
    }

    /// Split a micro-batch further when `count × paddedLength` would exceed the patch
    /// budget — the guard against one oversized image inflating the padded pixel tensor.
    private static func splitToPatchBudget(_ rows: [PreparedRow], budget: Int) -> [[PreparedRow]] {
        var slices: [[PreparedRow]] = []
        var current: [PreparedRow] = []
        var longest = 0
        for row in rows {
            let candidate = max(longest, row.grid?.product ?? 0)
            if !current.isEmpty && (current.count + 1) * candidate > budget {
                slices.append(current)
                current = []
                longest = 0
            }
            current.append(row)
            longest = max(longest, row.grid?.product ?? 0)
        }
        if !current.isEmpty { slices.append(current) }
        return slices
    }

    /// Right-pad a `[3, 1, seq]` position-id block with zeros (pad positions are never read
    /// by a real token under a causal mask).
    private static func padColumns(_ positions: MLXArray, to length: Int) -> MLXArray {
        let n = positions.dim(2)
        guard n < length else { return positions }
        return padded(
            positions,
            widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, length - n))],
            mode: .constant, value: MLXArray(Int32(0)))
    }

    /// Total image area, the sort key that keeps a micro-batch's images similarly sized.
    private static func imageArea(_ content: Qwen3VLContent) -> Int {
        content.images.reduce(0) { $0 + $1.width * $1.height }
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

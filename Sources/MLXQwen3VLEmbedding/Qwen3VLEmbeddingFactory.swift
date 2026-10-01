// Copyright © 2026
//
// A VLM model factory that loads the *vendored* ``Qwen3VLBackbone`` (which exposes
// `lastHiddenState`) for the `qwen3_vl` model type, while reusing MLXVLM's stock
// processor and model registries + the standard load path. Both heads load through it:
// the embedder pools the hidden state, the reranker projects it onto `W[yes] − W[no]`.

import Foundation
import MLXLMCommon
import MLXVLM

/// Pass-through tokenizer that suppresses auto-appended special tokens on `encode(text:)`.
///
/// This model's tokenizer appends `<|endoftext|>` (151643) when `addSpecialTokens` is true,
/// which breaks the stock `QwenVL.replacePaddingTokens` image-placeholder search (it looks for
/// `encode("<|vision_start|><|image_pad|><|vision_end|>")` as an exact subsequence, and the
/// spurious trailing token means it never matches). The chat-template path and our explicit
/// EOS-append for embedding pooling are unaffected (they don't go through this `encode`).
struct NoAutoSpecialsTokenizer: MLXLMCommon.Tokenizer {
    let base: any MLXLMCommon.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: false)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }
    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

/// Wraps another loader, returning a ``NoAutoSpecialsTokenizer``.
struct Qwen3VLEmbeddingTokenizerLoader: TokenizerLoader {
    let base: any TokenizerLoader
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        NoAutoSpecialsTokenizer(base: try await base.load(from: directory))
    }
}

/// Replica of MLXVLM's private `create` registry helper (decode config → init model).
private func embeddingCreate<C: Decodable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

enum Qwen3VLEmbeddingFactory {
    /// A factory whose `qwen3_vl` type creates the vendored hidden-state-exposing backbone.
    static let shared = VLMModelFactory(
        typeRegistry: ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3_vl": embeddingCreate(Qwen3VLConfiguration.self, Qwen3VLBackbone.init)
        ]),
        processorRegistry: VLMProcessorTypeRegistry.shared,
        modelRegistry: VLMRegistry.shared)
}

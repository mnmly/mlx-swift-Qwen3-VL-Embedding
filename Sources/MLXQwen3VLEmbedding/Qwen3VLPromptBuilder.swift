// Copyright © 2026
//
// Ports the reference repo's prompt assembly (`format_model_input` for the
// embedder, `format_mm_instruction` / `format_mm_content` for the reranker) into
// ``MLXLMCommon/UserInput`` values.
//
// Critical parity detail: the order of text / image items within a user message
// matters (the reranker interleaves `<Query>: → image → <Document>: → image`).
// The stock `Qwen3VLMessageGenerator` front-loads all images, which would break
// that ordering, so we emit raw `UserInput.Prompt.messages` dicts directly — the
// processor passes `.messages` through untouched (verified in
// `MLXLMCommon.MessageGenerator.generate(from:)`).

import CoreImage
import Foundation
import MLXLMCommon
import MLXVLM

enum Qwen3VLPromptBuilder {

    /// Reference: `instruction.strip()` then append `'.'` if the final character is
    /// not Unicode punctuation. Applied to embedder instructions only.
    static func normalizeInstruction(_ instruction: String) -> String {
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = trimmed.last else { return trimmed }
        return last.isPunctuation ? trimmed : trimmed + "."
    }

    // MARK: - Embedding (`format_model_input`)

    /// Build the embedder ``UserInput``: a system turn carrying the instruction and
    /// a user turn whose content is `[images…, texts…]` (videos deferred), or a
    /// single `"NULL"` text when empty.
    static func embeddingUserInput(
        content: Qwen3VLContent,
        instruction: String,
        minPixels: Int,
        maxPixels: Int
    ) -> UserInput {
        let system = message(role: "system", content: [textItem(normalizeInstruction(instruction))])

        var userContent: [Message] = []
        if content.isEmpty {
            userContent.append(textItem("NULL"))
        } else {
            userContent.append(contentsOf: content.images.map { _ in imageItem() })
            userContent.append(contentsOf: content.texts.map { textItem($0) })
        }
        let user = message(role: "user", content: userContent)

        return makeUserInput(
            messages: [system, user], images: content.images,
            minPixels: minPixels, maxPixels: maxPixels)
    }

    // MARK: - Reranking (`format_mm_instruction`)

    private static let rerankerSystemPrompt =
        "Judge whether the Document meets the requirements based on the Query and the "
        + "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."

    /// Build the reranker ``UserInput`` for one (query, document) pair: a fixed system
    /// turn and a user turn `[<Instruct>:…, <Query>: …query…, \n<Document>: …doc…]`.
    static func rerankUserInput(
        query: Qwen3VLContent,
        document: Qwen3VLContent,
        instruction: String,
        minPixels: Int,
        maxPixels: Int
    ) -> UserInput {
        let system = message(role: "system", content: [textItem(rerankerSystemPrompt)])

        var userContent: [Message] = [textItem("<Instruct>: " + instruction)]
        userContent.append(contentsOf: contentItems(query, prefix: "<Query>:"))
        userContent.append(contentsOf: contentItems(document, prefix: "\n<Document>:"))
        let user = message(role: "user", content: userContent)

        // Images appear query-first then document-first, matching content order.
        let images = query.images + document.images
        return makeUserInput(
            messages: [system, user], images: images, minPixels: minPixels, maxPixels: maxPixels)
    }

    /// Reference `format_mm_content`: `[prefix, images…, texts…]`, or `[prefix, "NULL"]`
    /// when the side carries nothing.
    private static func contentItems(_ content: Qwen3VLContent, prefix: String) -> [Message] {
        var items: [Message] = [textItem(prefix)]
        if content.isEmpty {
            items.append(textItem("NULL"))
        } else {
            items.append(contentsOf: content.images.map { _ in imageItem() })
            items.append(contentsOf: content.texts.map { textItem($0) })
        }
        return items
    }

    // MARK: - Dict helpers

    private static func textItem(_ text: String) -> Message {
        ["type": "text", "text": text]
    }

    private static func imageItem() -> Message {
        ["type": "image"]
    }

    private static func message(role: String, content: [Message]) -> Message {
        ["role": role, "content": content]
    }

    private static func makeUserInput(
        messages: [Message], images: [CGImage], minPixels _: Int, maxPixels _: Int
    ) -> UserInput {
        // The stock processor's image path renders pixels in CoreImage's linear working
        // space (asMLXArray uses colorSpace: nil), which darkens midtones vs the reference's
        // sRGB-encoded PIL pixels. Pre-applying the sRGB tone curve (as the video path does)
        // makes the rendered values match. See the image-preprocessing parity work.
        //
        // Note: the Qwen3-VL processor resizes from the model's `preprocessor_config`
        // (`config.size`), not from per-call `UserInput.Processing` overrides, so the
        // `minPixels`/`maxPixels` budgets are informational here.
        let imageInputs: [UserInput.Image] = images.map { .ciImage(CIImage(cgImage: $0).toSRGB()) }
        return UserInput(prompt: .messages(messages), images: imageInputs)
    }
}

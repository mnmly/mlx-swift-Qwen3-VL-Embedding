// Copyright © 2026
//
// Sendable description of one multimodal item (a query, a document, or an
// embedding input). Images are carried as `CGImage` (Sendable) so a value can
// cross into the engine actor; they're converted to `CIImage` only inside the
// model `perform` closure.

import CoreGraphics
import Foundation

/// A bundle of texts + images that forms one side of an embedding or rerank request.
///
/// Mirrors the flexible `text` / `image` arguments of the reference repo's
/// `format_model_input` / `format_mm_content` (video support is deferred).
public struct Qwen3VLContent: Sendable {
    public var texts: [String]
    public var images: [CGImage]

    public init(texts: [String] = [], images: [CGImage] = []) {
        self.texts = texts
        self.images = images
    }

    /// `true` when there is nothing to embed (reference emits a `"NULL"` text item).
    public var isEmpty: Bool { texts.isEmpty && images.isEmpty }

    public static func text(_ text: String) -> Qwen3VLContent {
        Qwen3VLContent(texts: [text])
    }

    public static func image(_ image: CGImage) -> Qwen3VLContent {
        Qwen3VLContent(images: [image])
    }

    public static func textImage(_ text: String, _ image: CGImage) -> Qwen3VLContent {
        Qwen3VLContent(texts: [text], images: [image])
    }
}

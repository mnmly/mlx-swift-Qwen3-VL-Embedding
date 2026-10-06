import CoreGraphics
import XCTest

@testable import MLXQwen3VLEmbedding

/// Phase 0 smoke: the package builds and links. Model-backed parity tests
/// (which need multi-GB weights) land in later phases and skip when weights
/// are absent.
final class ScaffoldTests: XCTestCase {
    func testVersionPresent() {
        XCTAssertFalse(MLXQwen3VLEmbedding.version.isEmpty)
    }

    func testTaskCases() {
        XCTAssertEqual(Qwen3VLTask.allCases.count, 2)
    }

    /// The reference resizes to its own 4–1800-token budget and skips the processor's resize, so
    /// the builder must hand the budget to the mlx-swift-lm processor; without it the processor
    /// falls back to the checkpoint's 1,280-token cap (large-image cosine 0.994 instead of 0.999).
    func testEmbeddingInputCarriesPixelBudget() throws {
        let ctx = try XCTUnwrap(CGContext(
            data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let image = try XCTUnwrap(ctx.makeImage())
        let config = Qwen3VLEmbeddingConfig(modelDirectory: URL(filePath: "/dev/null"), task: .embedding)
        let input = Qwen3VLPromptBuilder.embeddingUserInput(
            content: Qwen3VLContent(images: [image]), instruction: "Represent the user's input.",
            minPixels: config.minPixels, maxPixels: config.maxPixels)
        XCTAssertEqual(input.processing.minPixels, 4 * 32 * 32)
        XCTAssertEqual(input.processing.maxPixels, 1800 * 32 * 32)
    }
}

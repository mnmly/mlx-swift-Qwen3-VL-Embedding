import CoreGraphics
import Foundation
import MLX
import XCTest

@testable import MLXQwen3VLEmbedding

/// The batched image path must return the same vectors as embedding one image at a time.
///
/// The images are synthesized at deliberately mismatched sizes so every guard the batch has to
/// honour is exercised: different patch grids, different prompt lengths (so rows are
/// right-padded), and two distinct grids that happen to carry the same token count (which is
/// what makes the M-RoPE position cache need the grid in its key, not just the tokens).
///
/// The strict comparison runs in **float32**. In the checkpoint's own bfloat16 the two paths
/// select different Metal kernels, and 28 decoder layers amplify that rounding to ~1e-3 of the
/// pooled vector — but so does any other harmless reshuffle: measured against a float32
/// reference, unbatched bfloat16 lands at cosine 0.9987 and batched bfloat16 at 0.9987 as well.
/// Batching therefore costs no accuracy; asserting 0.9999 in bfloat16 would be asserting
/// something the model itself does not deliver.
///
/// Loads a multi-GB checkpoint, so it **skips** unless `QWEN3VL_EMBED_MODEL` is set or the
/// snapshot is in the Hugging Face cache.
final class BatchParityTests: XCTestCase {

    func testBatchedImageEmbeddingMatchesSingleInFloat32() async throws {
        let session = try await Self.loadSession(computeDType: .float32)
        let images = Self.testImages()

        var singles: [[Float]] = []
        for image in images { singles.append(try await session.embed(.image(image))) }
        let batched = try await session.embed(images: images, batchSize: 8)

        XCTAssertEqual(batched.count, singles.count)
        for (index, (single, batch)) in zip(singles, batched).enumerated() {
            XCTAssertGreaterThanOrEqual(
                ParityTests.cosine(single, batch), 0.9999,
                "image[\(index)] (\(images[index].width)×\(images[index].height)) batched vs single")
        }
    }

    /// The same check in the shipped bfloat16, at the model's own numerical resolution.
    func testBatchedImageEmbeddingMatchesSingleInBFloat16() async throws {
        let session = try await Self.loadSession(computeDType: nil)
        let images = Self.testImages()

        var singles: [[Float]] = []
        for image in images { singles.append(try await session.embed(.image(image))) }
        let batched = try await session.embed(images: images, batchSize: 8)

        for (index, (single, batch)) in zip(singles, batched).enumerated() {
            XCTAssertGreaterThanOrEqual(
                ParityTests.cosine(single, batch), 0.998,
                "image[\(index)] batched vs single (bfloat16 noise floor)")
        }
    }

    /// A batch mixing text-only and image items keeps every vector in input order.
    func testMixedBatchPreservesOrder() async throws {
        let session = try await Self.loadSession(computeDType: nil)
        let images = Self.testImages()
        let contents: [Qwen3VLContent] =
            [.text("a printed ornament"), .image(images[0]), .text("a diagram"), .image(images[1])]

        let batched = try await session.embed(contents, batchSize: 4)
        XCTAssertEqual(batched.count, contents.count)
        for (index, content) in contents.enumerated() {
            let single = try await session.embed(content)
            XCTAssertGreaterThanOrEqual(
                ParityTests.cosine(single, batched[index]), 0.998, "item[\(index)] order/parity")
        }
    }

    // MARK: - Helpers

    private static func loadSession(computeDType: DType?) async throws
        -> Qwen3VLEmbeddingSession
    {
        try await Qwen3VLEmbeddingSession.load(
            .init(
                modelDirectory: URL(filePath: try modelDirectory()), task: .embedding,
                computeDType: computeDType))
    }

    /// Sizes chosen for what they do to the patch grid (`round(side / 32) * 32`, then `/ 16`):
    /// varied aspect ratios and lengths, plus 704×1280 and 1408×640 — 44×80 and 88×40, two
    /// different grids with the same 3520-patch count.
    private static func testImages() -> [CGImage] {
        [(320, 256), (704, 1280), (1408, 640), (512, 384), (960, 288), (256, 960), (448, 448)]
            .enumerated()
            .compactMap { seed, size in gradientImage(width: size.0, height: size.1, seed: seed) }
    }

    /// A deterministic, non-uniform RGB image — flat colour would make every patch identical
    /// and hide an ordering mistake.
    private static func gradientImage(width: Int, height: Int, seed: Int) -> CGImage? {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let offset = (y * width + x) * 4
                pixels[offset] = UInt8((x &* 7 &+ y &* 3 &+ seed &* 41) % 256)
                pixels[offset + 1] = UInt8((x &* 3 &+ y &* 11 &+ seed &* 97) % 256)
                pixels[offset + 2] = UInt8((x &* 13 &+ y &* 5 &+ seed &* 17) % 256)
                pixels[offset + 3] = 255
            }
        }
        let space = CGColorSpaceCreateDeviceRGB()
        guard
            let provider = CGDataProvider(data: Data(pixels) as CFData),
            let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        return image
    }

    /// The embedding snapshot: `QWEN3VL_EMBED_MODEL` if set, else the Hugging Face cache.
    private static func modelDirectory() throws -> String {
        if let value = ProcessInfo.processInfo.environment["QWEN3VL_EMBED_MODEL"], !value.isEmpty {
            return value
        }
        let repoRoot = FileManager.default.homeDirectoryForCurrentUser
            .appending(
                components: ".cache", "huggingface", "hub",
                "models--Qwen--Qwen3-VL-Embedding-2B")
        let resolved = Qwen3VLModel.resolveSnapshotDirectory(repoRoot)
        guard FileManager.default.fileExists(atPath: resolved.appending(component: "config.json").path)
        else {
            throw XCTSkip("set QWEN3VL_EMBED_MODEL or download Qwen/Qwen3-VL-Embedding-2B")
        }
        return resolved.path
    }
}

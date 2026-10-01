import CoreGraphics
import Foundation
import MLX
import XCTest

@testable import MLXQwen3VLEmbedding

/// The reranker's float32 score head and its batched text path.
///
/// Batched and one-at-a-time scoring run the same math on differently shaped matrices, so they
/// agree to float32 rounding in float32 and to the checkpoint's own noise floor in bfloat16
/// (both bfloat16 paths sit ~0.025 from a float32 run of the same model; batching adds nothing
/// on top of that).
///
/// Loads a multi-GB checkpoint, so it **skips** unless `QWEN3VL_RERANK_MODEL` is set or the
/// snapshot is in the Hugging Face cache.
final class RerankBatchTests: XCTestCase {

    private static let query = "What did the author argue about machine intelligence?"

    /// Documents of deliberately different lengths, so every micro-batch is right-padded.
    private static let documents: [String] = [
        "Machines cannot think.",
        "The author argued that intelligent behaviour depends on a background of everyday "
            + "coping that cannot be captured in explicit rules, and that symbolic AI therefore "
            + "rested on a mistaken picture of the mind.",
        "Bananas are a good source of potassium.",
        "In the third chapter the author turns to the history of the field, describing early "
            + "optimism about chess programs, language translation and pattern recognition, and "
            + "the gradual realisation that progress on toy problems did not carry over to the "
            + "open-ended situations people handle without effort. The argument is not that "
            + "computers are useless but that the rule-following model of cognition is wrong.",
        "Paris is the capital of France.",
        "Heidegger describes equipment as ready-to-hand.",
        "A short note on typesetting conventions used in this edition.",
    ]

    func testBatchedRerankMatchesSingleInFloat32() async throws {
        let session = try await Self.loadSession(computeDType: .float32)
        let single = try await session.rerank(
            query: Self.query, documents: Self.documents, batchSize: 1)
        for batchSize in [3, 8] {
            let batched = try await session.rerank(
                query: Self.query, documents: Self.documents, batchSize: batchSize)
            XCTAssertEqual(batched.count, single.count)
            for (index, (a, b)) in zip(single, batched).enumerated() {
                XCTAssertEqual(a, b, accuracy: 0.003, "doc[\(index)] batch \(batchSize) vs single")
            }
        }
    }

    /// An image document takes the one-at-a-time path with the same float32 head, and the text
    /// documents around it keep their scores and their order.
    func testImageDocumentInATextBatch() async throws {
        let session = try await Self.loadSession(computeDType: .float32)
        let image = try XCTUnwrap(Self.gradientImage(width: 320, height: 256))
        let texts = Array(Self.documents.prefix(4))
        let textOnly = try await session.rerank(query: Self.query, documents: texts)

        var mixed: [Qwen3VLContent] = texts.map { .text($0) }
        mixed.insert(.image(image), at: 2)
        let scores = try await session.rerank(query: .text(Self.query), documents: mixed)

        XCTAssertEqual(scores.count, mixed.count)
        XCTAssert(scores[2] > 0 && scores[2] < 1, "image score \(scores[2])")
        let textScores = scores.enumerated().filter { $0.offset != 2 }.map(\.element)
        for (index, (a, b)) in zip(textOnly, textScores).enumerated() {
            XCTAssertEqual(a, b, accuracy: 0.003, "text doc[\(index)] beside an image")
        }
    }

    func testBatchedRerankMatchesSingleInBFloat16() async throws {
        let session = try await Self.loadSession(computeDType: nil)
        let single = try await session.rerank(
            query: Self.query, documents: Self.documents, batchSize: 1)
        let batched = try await session.rerank(query: Self.query, documents: Self.documents)
        for (index, (a, b)) in zip(single, batched).enumerated() {
            XCTAssertEqual(a, b, accuracy: 0.05, "doc[\(index)] batched vs single (bf16 noise floor)")
        }
    }

    /// Regression: reading `logits[yes] − logits[no]` from the bfloat16 vocabulary projection
    /// put every score's logit on a 1/16 grid. The float32 head must not.
    func testScoresAreNotOnTheBFloat16LogitGrid() async throws {
        let session = try await Self.loadSession(computeDType: nil)
        let scores = try await session.rerank(query: Self.query, documents: Self.documents)
        let offGrid = scores.filter { score in
            let logit = log(Double(score) / (1 - Double(score))) * 16
            return abs(logit - logit.rounded()) > 0.01
        }
        XCTAssertGreaterThanOrEqual(offGrid.count, scores.count - 1, "scores \(scores)")
        XCTAssertEqual(Set(scores).count, scores.count, "no ties among distinct documents")
    }

    // MARK: - Helpers

    /// A deterministic, non-uniform RGB image.
    private static func gradientImage(width: Int, height: Int) -> CGImage? {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let offset = (y * width + x) * 4
                pixels[offset] = UInt8((x &* 7 &+ y &* 3) % 256)
                pixels[offset + 1] = UInt8((x &* 3 &+ y &* 11) % 256)
                pixels[offset + 2] = UInt8((x &* 13 &+ y &* 5) % 256)
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    private static func loadSession(computeDType: DType?) async throws -> Qwen3VLEmbeddingSession {
        try await Qwen3VLEmbeddingSession.load(
            .init(
                modelDirectory: URL(filePath: try modelDirectory()), task: .reranker,
                computeDType: computeDType))
    }

    private static func modelDirectory() throws -> String {
        if let value = ProcessInfo.processInfo.environment["QWEN3VL_RERANK_MODEL"], !value.isEmpty {
            return value
        }
        let repoRoot = FileManager.default.homeDirectoryForCurrentUser
            .appending(
                components: ".cache", "huggingface", "hub", "models--Qwen--Qwen3-VL-Reranker-2B")
        let resolved = Qwen3VLModel.resolveSnapshotDirectory(repoRoot)
        guard FileManager.default.fileExists(atPath: resolved.appending(component: "config.json").path)
        else {
            throw XCTSkip("set QWEN3VL_RERANK_MODEL or download Qwen/Qwen3-VL-Reranker-2B")
        }
        return resolved.path
    }
}

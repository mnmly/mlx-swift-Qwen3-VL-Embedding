import CoreGraphics
import Foundation
import ImageIO
import XCTest

@testable import MLXQwen3VLEmbedding

/// End-to-end parity against the PyTorch/transformers reference, using fixtures produced by
/// `Tools/gen_fixtures.py` (`Tests/Fixtures/parity.json`).
///
/// These load multi-GB checkpoints, so they **skip** unless the model paths are provided via
/// environment variables:
///   - `QWEN3VL_EMBED_MODEL`  — a Qwen3-VL-Embedding snapshot directory
///   - `QWEN3VL_RERANK_MODEL` — a Qwen3-VL-Reranker snapshot directory
final class ParityTests: XCTestCase {

    func testRerankerParity() async throws {
        let fixtures = try loadFixtures()
        let modelPath = try modelDirectory(
            env: "QWEN3VL_RERANK_MODEL", hfRepo: "Qwen/Qwen3-VL-Reranker-2B")

        let session = try await Qwen3VLEmbeddingSession.load(
            .init(modelDirectory: URL(filePath: modelPath), task: .reranker))
        let scores = try await session.rerank(
            query: fixtures.inputs.rerank.query, documents: fixtures.inputs.rerank.documents)

        XCTAssertEqual(scores.count, fixtures.reference.rerankScores.count)
        for (got, ref) in zip(scores, fixtures.reference.rerankScores) {
            XCTAssertEqual(got, ref, accuracy: 0.02, "reranker score parity")
        }
    }

    func testEmbeddingTextParity() async throws {
        let fixtures = try loadFixtures()
        let modelPath = try modelDirectory(
            env: "QWEN3VL_EMBED_MODEL", hfRepo: "Qwen/Qwen3-VL-Embedding-2B")

        let session = try await Qwen3VLEmbeddingSession.load(
            .init(modelDirectory: URL(filePath: modelPath), task: .embedding))
        let texts = fixtures.inputs.embed.embedTexts
        let vectors = try await session.embed(texts: texts)

        XCTAssertEqual(vectors.count, texts.count)
        for (i, vec) in vectors.enumerated() {
            let ref = fixtures.reference.embedVectors[i]
            XCTAssertGreaterThanOrEqual(
                Self.cosine(vec, ref), 0.99, "text[\(i)] embedding cosine vs reference")
        }
    }

    func testEmbeddingImageParity() async throws {
        let fixtures = try loadFixtures()
        guard let imagePath = fixtures.inputs.embed.embedImage else {
            throw XCTSkip("fixtures have no embed_image; regenerate with --image to run this test")
        }
        guard let cg = Self.loadCGImage(imagePath) else {
            throw XCTSkip("could not load fixture image at \(imagePath)")
        }
        let modelPath = try modelDirectory(
            env: "QWEN3VL_EMBED_MODEL", hfRepo: "Qwen/Qwen3-VL-Embedding-2B")

        let session = try await Qwen3VLEmbeddingSession.load(
            .init(modelDirectory: URL(filePath: modelPath), task: .embedding))
        let vector = try await session.embed(.image(cg))

        // The image reference is the last embed_vectors entry (texts first, then image).
        let ref = fixtures.reference.embedVectors[fixtures.inputs.embed.embedTexts.count]
        XCTAssertGreaterThanOrEqual(
            Self.cosine(vector, ref), 0.99, "image embedding cosine vs reference")
    }

    private static func loadCGImage(_ path: String) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(URL(filePath: path) as CFURL, nil) else {
            return nil
        }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    // MARK: - Helpers

    struct Fixtures: Decodable {
        struct Inputs: Decodable {
            struct Rerank: Decodable {
                let query: String
                let documents: [String]
            }
            struct Embed: Decodable {
                let embedTexts: [String]
                let embedImage: String?
            }
            let rerank: Rerank
            let embed: Embed
        }
        struct Reference: Decodable {
            let rerankScores: [Float]
            let embedVectors: [[Float]]
        }
        let inputs: Inputs
        let reference: Reference
    }

    private func loadFixtures(file: StaticString = #filePath) throws -> Fixtures {
        // Tests/MLXQwen3VLEmbeddingTests/ParityTests.swift -> Tests/Fixtures/parity.json
        let url = URL(filePath: "\(file)")
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(components: "Fixtures", "parity.json")
        guard let data = try? Data(contentsOf: url) else {
            throw XCTSkip("parity fixtures missing — run Tools/gen_fixtures.py to create \(url.path)")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Fixtures.self, from: data)
    }

    /// Resolve a model directory: the `env` override if set, else the standard Hugging Face
    /// cache (`~/.cache/huggingface/hub/models--<org>--<name>`). Skips if neither is present
    /// (the multi-GB weights are not in CI).
    private func modelDirectory(env: String, hfRepo: String) throws -> String {
        if let value = ProcessInfo.processInfo.environment[env], !value.isEmpty { return value }

        let cacheName = "models--" + hfRepo.replacingOccurrences(of: "/", with: "--")
        let repoRoot = FileManager.default.homeDirectoryForCurrentUser
            .appending(components: ".cache", "huggingface", "hub", cacheName)
        if FileManager.default.fileExists(atPath: repoRoot.path) {
            let resolved = Qwen3VLModel.resolveSnapshotDirectory(repoRoot)
            if FileManager.default.fileExists(
                atPath: resolved.appending(component: "config.json").path)
            {
                return resolved.path
            }
        }
        throw XCTSkip("set \(env) or download \(hfRepo) to run this parity test")
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< min(a.count, b.count) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom > 0 ? dot / denom : 0
    }
}

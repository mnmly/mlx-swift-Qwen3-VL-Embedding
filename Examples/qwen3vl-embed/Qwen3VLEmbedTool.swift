// qwen3vl-embed — CLI frontend over ``Qwen3VLEmbeddingSession``.
//
// Thin: parse args, load the shared Session, run a head, print. All model and
// prompt logic lives in the library.

import ArgumentParser
import Foundation
import ImageIO
import MLXQwen3VLEmbedding

@main
struct Qwen3VLEmbedTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qwen3vl-embed",
        abstract: "Embed / rerank multimodal inputs with Qwen3-VL on Apple Silicon.",
        version: MLXQwen3VLEmbedding.version,
        subcommands: [Embed.self, Rerank.self, Bench.self]
    )
}

/// Throughput + memory-leak harness: load the model once, run the pipeline in a loop, and watch
/// MLX's active memory. Flat `active` across iterations ⇒ no leak (the large `peak` is MLX's
/// reusable buffer cache, not a leak). Build Release for representative numbers.
struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmark embed/rerank throughput and check for memory leaks.")

    @Option(name: .shortAndLong, help: "Path to a Qwen3-VL snapshot directory.")
    var model: String

    @Option(help: "Which head to benchmark: embedding or reranker.")
    var task: Qwen3VLTask = .embedding

    @Option(help: "Measured iterations.")
    var iterations: Int = 20

    @Option(help: "Warmup iterations (excluded from timing).")
    var warmup: Int = 3

    @Argument(help: "Text to embed / use as the rerank query.")
    var text: String = "A photograph of a golden retriever puppy."

    func run() async throws {
        let session = try await Qwen3VLEmbeddingSession.load(
            .init(modelDirectory: URL(filePath: model), task: task))

        @Sendable func once() async throws {
            switch task {
            case .embedding: _ = try await session.embed(texts: [text])
            case .reranker: _ = try await session.rerank(query: text, documents: [text])
            }
        }

        for _ in 0 ..< warmup { try await once() }

        let baseline = Qwen3VLEmbeddingSession.memorySnapshot()
        let clock = ContinuousClock()
        var samples: [Double] = []
        samples.reserveCapacity(iterations)

        for i in 0 ..< iterations {
            let elapsed = try await clock.measure { try await once() }
            samples.append(elapsed.seconds)
            if i % 5 == 0 || i == iterations - 1 {
                let m = Qwen3VLEmbeddingSession.memorySnapshot()
                print(
                    String(
                        format: "iter %2d  %.3fs  active=%@ (Δ%@)  cache=%@  peak=%@",
                        i, samples[i], mb(m.active), mb(m.active - baseline.active),
                        mb(m.cache), mb(m.peak)))
            }
        }

        let sorted = samples.sorted()
        let mean = samples.reduce(0, +) / Double(samples.count)
        let median = sorted[sorted.count / 2]
        let final = Qwen3VLEmbeddingSession.memorySnapshot()
        let leaked = final.active - baseline.active
        print(
            String(
                format: "\n%@  mean=%.3fs  median=%.3fs  (%.1f/s)\nactive drift=%@  ⇒ %@",
                task.rawValue, mean, median, 1.0 / mean, mb(leaked),
                abs(leaked) < 16 * 1_048_576 ? "no leak (flat active memory)" : "INVESTIGATE: active grew"))
    }

    private func mb(_ bytes: Int) -> String { String(format: "%.0fMB", Double(bytes) / 1_048_576) }
}

extension Qwen3VLTask: ExpressibleByArgument {}

extension Duration {
    fileprivate var seconds: Double {
        let c = components
        return Double(c.seconds) + Double(c.attoseconds) / 1e18
    }
}

/// Embed one or more texts and print the vectors as JSON `[[Float]]` (for parity checks).
struct Embed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Embed texts with Qwen3-VL-Embedding; prints JSON [[Float]].")

    @Option(name: .shortAndLong, help: "Path to a Qwen3-VL-Embedding snapshot directory.")
    var model: String

    @Option(help: "Matryoshka (MRL) output dimension; omit for the full vector.")
    var dim: Int?

    @Flag(help: "Skip L2-normalization (vectors are normalized by default).")
    var raw: Bool = false

    @Option(help: "Optional image file to embed as an extra (image-only) item, after the texts.")
    var image: String?

    @Argument(parsing: .remaining, help: "Zero or more texts to embed.")
    var texts: [String] = []

    func run() async throws {
        var contents = texts.map { Qwen3VLContent.text($0) }
        if let image {
            guard let cg = Self.loadCGImage(image) else {
                throw ValidationError("Could not load image: \(image)")
            }
            contents.append(.image(cg))
        }
        guard !contents.isEmpty else { throw ValidationError("Provide at least one text or --image.") }

        let config = Qwen3VLEmbeddingConfig(
            modelDirectory: URL(filePath: model), task: .embedding,
            normalize: !raw, embeddingDimension: dim)
        let session = try await Qwen3VLEmbeddingSession.load(config)
        let vectors = try await session.embed(contents)
        let data = try JSONSerialization.data(withJSONObject: vectors)
        print(String(decoding: data, as: UTF8.self))
    }

    static func loadCGImage(_ path: String) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(URL(filePath: path) as CFURL, nil) else {
            return nil
        }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

/// Score one query against several text documents and print them best-first.
struct Rerank: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Rank documents by relevance to a query (Qwen3-VL-Reranker).")

    @Option(name: .shortAndLong, help: "Path to a Qwen3-VL-Reranker snapshot directory.")
    var model: String

    @Option(help: "Task instruction injected into the system prompt.")
    var instruction: String?

    @Argument(help: "The search query.")
    var query: String

    @Argument(parsing: .remaining, help: "One or more candidate documents.")
    var documents: [String]

    func run() async throws {
        guard !documents.isEmpty else {
            throw ValidationError("Provide at least one document to rank.")
        }
        let config = Qwen3VLEmbeddingConfig(
            modelDirectory: URL(filePath: model),
            task: .reranker,
            instruction: instruction)

        let session = try await Qwen3VLEmbeddingSession.load(config)
        let ranked = try await session.rankedDocuments(
            query: .text(query), documents: documents.map { .text($0) })

        for entry in ranked {
            print(String(format: "%.4f  [%d] %@", entry.score, entry.index, documents[entry.index]))
        }
    }
}

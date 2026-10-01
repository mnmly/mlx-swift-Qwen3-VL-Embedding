// rerank-bench — latency sweep + batch parity for text reranking.
//
// Reads a JSON array of `{"id", "query", "documents"}` items, scores each query against its
// first `depth` documents at every requested batch size, and reports per-query latency and how
// far each batch size's scores sit from the first one's. `--scores` writes the raw scores so
// they can be compared against a reference implementation offline.

import ArgumentParser
import Foundation
import MLX
import MLXQwen3VLEmbedding

struct RerankBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rerank-bench",
        abstract: "Sweep text-reranking latency over batch sizes and verify batch parity.")

    struct Item: Decodable {
        let id: String
        let query: String
        let documents: [String]
    }

    @Option(name: .shortAndLong, help: "Path to a Qwen3-VL-Reranker snapshot directory.")
    var model: String

    @Option(help: "JSON file: an array of {id, query, documents}.")
    var input: String

    @Option(help: "How many queries to run (taken in file order); 0 runs all.")
    var queries: Int = 0

    @Option(help: "Comma-separated candidate depths to time.")
    var depths: String = "20,50"

    @Option(help: "Comma-separated batch sizes to sweep; the first is the parity baseline.")
    var batchSizes: String = "1,4,8,16"

    @Option(help: "Write {batchSize: {id: scores}} at the largest depth to this JSON file.")
    var scores: String?

    @Flag(help: "Run in float32 instead of the checkpoint's bfloat16.")
    var fp32: Bool = false

    func run() async throws {
        var items = try JSONDecoder().decode(
            [Item].self, from: Data(contentsOf: URL(filePath: input)))
        if queries > 0 { items = Array(items.prefix(queries)) }
        guard let first = items.first else { throw ValidationError("No items in \(input).") }
        let depthList = depths.split(separator: ",").compactMap { Int($0) }
        let sizes = batchSizes.split(separator: ",").compactMap { Int($0) }
        let maxDepth = depthList.max() ?? 50

        let session = try await Qwen3VLEmbeddingSession.load(
            .init(
                modelDirectory: URL(filePath: model), task: .reranker,
                computeDType: fp32 ? .float32 : nil))

        print("\(items.count) queries, depths \(depthList), batch sizes \(sizes)")
        print("  batch  depth   median s   mean s    max s   peak GPU")
        var allScores: [Int: [String: [Float]]] = [:]
        for size in sizes {
            // Warm the per-shape Metal pipelines so the first query doesn't pay compilation.
            _ = try await session.rerank(
                query: first.query, documents: Array(first.documents.prefix(maxDepth)),
                batchSize: size)
            for depth in depthList.sorted() {
                MLX.Memory.clearCache()
                MLX.Memory.peakMemory = 0
                var seconds: [Double] = []
                for item in items {
                    let docs = Array(item.documents.prefix(depth))
                    let start = Date()
                    let result = try await session.rerank(
                        query: item.query, documents: docs, batchSize: size)
                    seconds.append(-start.timeIntervalSinceNow)
                    if depth == maxDepth { allScores[size, default: [:]][item.id] = result }
                }
                let sorted = seconds.sorted()
                print(
                    String(
                        format: "  %5d  %5d   %8.3f  %7.3f  %7.3f   %6.2f GB",
                        size, depth, sorted[sorted.count / 2],
                        seconds.reduce(0, +) / Double(seconds.count), sorted.last ?? 0,
                        Double(MLX.Memory.peakMemory) / 1_073_741_824))
            }
        }

        if let base = sizes.first, let reference = allScores[base] {
            print("\nparity vs batch \(base) (depth \(maxDepth)):")
            for size in sizes.dropFirst() {
                var worst: Float = 0
                for (id, ref) in reference {
                    for (a, b) in zip(ref, allScores[size]?[id] ?? []) { worst = max(worst, abs(a - b)) }
                }
                print(String(format: "  batch %3d  max |Δscore| %.6f", size, worst))
            }
        }

        if let scores {
            let payload = Dictionary(
                uniqueKeysWithValues: allScores.map { (String($0.key), $0.value) })
            try JSONSerialization.data(withJSONObject: payload).write(to: URL(filePath: scores))
            print("wrote \(scores)")
        }
    }
}

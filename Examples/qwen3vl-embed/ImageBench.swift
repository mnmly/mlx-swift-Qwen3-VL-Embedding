// image-bench — throughput sweep + parity check for batched image embedding.
//
// Answers the two questions batching raises: does a micro-batch produce the same vector as
// the one-at-a-time path (parity), and how much faster is it (sweep). Images are decoded up
// front so the numbers measure the model, not ImageIO.

import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXQwen3VLEmbedding

struct ImageBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image-bench",
        abstract: "Sweep image-embedding throughput over batch sizes and verify batch parity.")

    @Option(name: .shortAndLong, help: "Path to a Qwen3-VL-Embedding snapshot directory.")
    var model: String

    @Option(help: "Directory of images to embed.")
    var images: String

    @Option(help: "How many images to time (taken in sorted order).")
    var count: Int = 128

    @Option(help: "Comma-separated batch sizes to sweep; 1 means the unbatched path.")
    var batchSizes: String = "1,4,8,16,32"

    @Option(help: "Images embedded once before timing, to warm the model.")
    var warmup: Int = 4

    @Option(help: "Timed passes per batch size; the fastest is reported.")
    var repeats: Int = 3

    @Option(help: "Images to check batched-vs-single parity on; 0 skips the check.")
    var parity: Int = 32

    @Option(help: "Batch size used for the parity check.")
    var parityBatch: Int = 8

    @Option(help: "Longest edge to decode to; 0 keeps the original resolution.")
    var decodeMaxEdge: Int = 0

    @Option(help: "Write the parity vectors to this JSON file (for cross-run comparison).")
    var dump: String?

    @Flag(help: "Run in float32 instead of the checkpoint's bfloat16 (parity reference).")
    var fp32: Bool = false

    func run() async throws {
        let urls = try Self.imageURLs(in: images, limit: (max(count, parity) + warmup) * 2)
        guard urls.count >= max(count, parity) + warmup else {
            throw ValidationError(
                "Need \(max(count, parity) + warmup) images in \(images), found \(urls.count).")
        }

        let decodeStart = Date()
        // The Qwen3-VL processor rejects an image whose side is below patch × merge (32).
        let decoded = urls.compactMap { Self.loadCGImage($0, maxEdge: decodeMaxEdge) }
            .filter { $0.width >= 32 && $0.height >= 32 }
        let decodeSeconds = -decodeStart.timeIntervalSinceNow
        print(
            String(
                format: "decoded %d images in %.2fs (%.1f/s, decode only)",
                decoded.count, decodeSeconds, Double(decoded.count) / decodeSeconds))

        let session = try await Qwen3VLEmbeddingSession.load(
            .init(
                modelDirectory: URL(filePath: model), task: .embedding,
                computeDType: fp32 ? .float32 : nil))

        let warm = Array(decoded.prefix(warmup))
        for image in warm { _ = try await session.embed(.image(image)) }

        let pool = Array(decoded.dropFirst(warmup))

        if parity > 0 {
            let subject = Array(pool.prefix(parity))
            print("\nparity: \(subject.count) images, batch \(parityBatch) vs single")
            var singles: [[Float]] = []
            for image in subject { singles.append(try await session.embed(.image(image))) }
            let batched = try await session.embed(images: subject, batchSize: parityBatch)

            var cosines: [Float] = []
            for (a, b) in zip(singles, batched) { cosines.append(Self.cosine(a, b)) }
            let worst = cosines.min() ?? 0
            let mean = cosines.reduce(0, +) / Float(cosines.count)
            // In float32 the two paths are the same computation and must agree to 0.9999. In
            // the checkpoint's bfloat16 they pick different Metal kernels, and 28 decoder
            // layers amplify that rounding — but no more than the checkpoint's own
            // quantization does (both paths sit at ~0.9987 against a float32 reference), so
            // the bar there is the noise floor, not bit-parity. Use --fp32 for the strict run.
            let bar: Float = fp32 ? 0.9999 : 0.998
            print(
                String(
                    format: "  min cosine %.6f   mean %.6f   %@ (bar %.4f, %@)",
                    worst, mean, worst >= bar ? "PASS" : "FAIL", bar,
                    fp32 ? "float32" : "bfloat16 noise floor"))
            if let dump {
                let payload: [String: [[Float]]] = ["single": singles, "batched": batched]
                try JSONSerialization.data(withJSONObject: payload)
                    .write(to: URL(filePath: dump))
                print("  wrote \(dump)")
            }
            for (offset, cosine) in cosines.enumerated().sorted(by: { $0.element < $1.element })
                .prefix(6)
            {
                let image = subject[offset]
                print(
                    String(format: "    %.6f  %4d×%-4d", cosine, image.width, image.height))
            }
        }

        let subject = Array(pool.prefix(count))
        print("\nthroughput: \(subject.count) images")
        print("  batch    seconds     img/s   speedup   peak GPU")
        var baseline: Double?
        for size in batchSizes.split(separator: ",").compactMap({ Int($0) }) {
            // Metal specializes pipelines per shape, so a batch size's first pass pays
            // compilation the later ones don't; report the fastest pass.
            var best = Double.infinity
            var peak = 0
            for _ in 0 ..< max(1, repeats) {
                MLX.Memory.clearCache()
                let start = Date()
                if size <= 1 {
                    for image in subject { _ = try await session.embed(.image(image)) }
                } else {
                    // Feed in windows so preprocessing memory stays bounded, as a corpus
                    // indexer would.
                    var cursor = 0
                    while cursor < subject.count {
                        let end = min(cursor + max(size * 4, 32), subject.count)
                        _ = try await session.embed(
                            images: Array(subject[cursor ..< end]), batchSize: size)
                        cursor = end
                    }
                }
                best = min(best, -start.timeIntervalSinceNow)
                peak = max(peak, MLX.Memory.snapshot().peakMemory)
            }
            let rate = Double(subject.count) / best
            if baseline == nil { baseline = rate }
            print(
                String(
                    format: "  %5d   %8.2f  %8.2f   %6.2fx   %6.2f GB",
                    size, best, rate, rate / (baseline ?? rate),
                    Double(peak) / 1_073_741_824))
        }
    }

    // MARK: - Helpers

    static func imageURLs(in directory: String, limit: Int) throws -> [URL] {
        let root = URL(filePath: directory)
        let names = try FileManager.default.contentsOfDirectory(atPath: root.path)
            .filter { ["jpg", "jpeg", "png"].contains(URL(filePath: $0).pathExtension.lowercased()) }
            .sorted()
        return names.prefix(limit).map { root.appending(component: $0) }
    }

    static func loadCGImage(_ url: URL, maxEdge: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        guard maxEdge > 0 else { return CGImageSourceCreateImageAtIndex(source, 0, nil) }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maxEdge,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< min(a.count, b.count) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denominator = na.squareRoot() * nb.squareRoot()
        return denominator > 0 ? dot / denominator : 0
    }
}

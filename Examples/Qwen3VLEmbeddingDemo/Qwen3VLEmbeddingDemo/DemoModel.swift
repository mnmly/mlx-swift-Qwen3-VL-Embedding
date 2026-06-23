// The GUI's view-model. It owns *only* presentation state + the cadence of driving the
// shared ``Qwen3VLEmbeddingSession``; all model/engine logic lives in the library. The CLI
// (`qwen3vl-embed`) drives the same Session — this is the swift-cli-gui-shared-driver pattern.

import CoreGraphics
import Foundation
import ImageIO
import MLXQwen3VLEmbedding
import Observation

@MainActor
@Observable
final class DemoModel {
    // Model locations. A folder picked via the sandbox Finder dialog (outside the app container)
    // is persisted as a security-scoped bookmark; a downloaded model lives in the app container.
    var embeddingModelURL: URL?
    var rerankerModelURL: URL?
    var embeddingRepoId = "Qwen/Qwen3-VL-Embedding-2B"
    var rerankerRepoId = "Qwen/Qwen3-VL-Reranker-2B"

    // Download state (shared by both models — one download at a time).
    var downloadProgress: Double?
    var downloadStatus = ""

    // Inputs (one item per line for the text blobs)
    var textsText = """
        A photograph of a golden retriever puppy.
        The Eiffel Tower at sunset.
        Quarterly revenue grew 12% year over year.
        """
    var query = "What is the capital of France?"
    var documentsText = """
        Paris is the capital of France.
        The Great Wall of China is a famous landmark.
        Bananas are a good source of potassium.
        """
    /// Images to embed alongside the texts (cross-modal similarity). Decoded eagerly when picked
    /// (while the sandbox grants access to the file) and kept in memory, so no later file read is
    /// needed — picked URLs are only transiently accessible.
    var images: [(name: String, image: CGImage)] = []

    var texts: [String] { Self.lines(textsText) }
    var documents: [String] { Self.lines(documentsText) }

    // Outputs
    var status = "Idle"
    var isBusy = false
    var similarity: [[Float]] = []
    var itemLabels: [String] = []
    var ranked: [(index: Int, score: Float)] = []
    var memory = "—"

    private var embedder: Qwen3VLEmbeddingSession?
    private var reranker: Qwen3VLEmbeddingSession?

    private static let embedBookmarkKey = "qwen3vl.embedding.bookmark"
    private static let rerankBookmarkKey = "qwen3vl.reranker.bookmark"

    init() {
        embeddingModelURL = Self.resolveBookmark(Self.embedBookmarkKey)
        rerankerModelURL = Self.resolveBookmark(Self.rerankBookmarkKey)
    }

    // MARK: - Model selection (Finder dialog + bookmark, or download)

    /// Use a folder the user picked in the Finder dialog (creates a persistent security-scoped
    /// bookmark) or one returned by a download.
    func setEmbeddingModel(_ url: URL) {
        persistBookmark(url, key: Self.embedBookmarkKey)
        embeddingModelURL = url
        embedder = nil
    }

    func setRerankerModel(_ url: URL) {
        persistBookmark(url, key: Self.rerankBookmarkKey)
        rerankerModelURL = url
        reranker = nil
    }

    func downloadEmbeddingModel() {
        download(repoId: embeddingRepoId) { [weak self] in self?.setEmbeddingModel($0) }
    }

    func downloadRerankerModel() {
        download(repoId: rerankerRepoId) { [weak self] in self?.setRerankerModel($0) }
    }

    private func download(repoId: String, assign: @escaping (URL) -> Void) {
        guard downloadProgress == nil, !isBusy else { return }
        downloadProgress = 0
        downloadStatus = "Downloading \(repoId)…"
        Task {
            do {
                let url = try await Qwen3VLEmbeddingSession.downloadModel(repoId: repoId) { frac in
                    Task { @MainActor in self.downloadProgress = frac }
                }
                assign(url)
                downloadStatus = "Downloaded \(repoId)"
            } catch {
                downloadStatus = "Download failed: \(error.localizedDescription)"
            }
            downloadProgress = nil
        }
    }

    private func persistBookmark(_ url: URL, key: String) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        if let data = try? url.bookmarkData(
            options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
        {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    private static func resolveBookmark(_ key: String) -> URL? {
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        var stale = false
        return try? URL(
            resolvingBookmarkData: data, options: [.withSecurityScope], relativeTo: nil,
            bookmarkDataIsStale: &stale)
    }

    // MARK: - Embedding / reranking

    func embed() {
        let texts = self.texts
        let imgs = images
        guard !(texts.isEmpty && imgs.isEmpty), !isBusy, let modelURL = embeddingModelURL else { return }
        run("Embedding…") { [self] in
            let session = try await loadEmbedder(modelURL)
            // Texts first, then images — one shared space, so the matrix shows cross-modal matches.
            var contents = texts.map { Qwen3VLContent.text($0) }
            var labels = texts.map { String($0.prefix(28)) }
            for img in imgs {
                contents.append(.image(img.image))
                labels.append("🖼 " + img.name)
            }
            let vectors = try await session.embed(contents)
            let sim = Self.cosineMatrix(vectors)
            await MainActor.run {
                self.similarity = sim
                self.itemLabels = labels
                self.memory = Self.formatMemory()
            }
        }
    }

    func rerank() {
        let documents = self.documents
        guard !query.isEmpty, !documents.isEmpty, !isBusy, let modelURL = rerankerModelURL else { return }
        let query = query
        run("Reranking…") { [self] in
            let session = try await loadReranker(modelURL)
            let ranked = try await session.rankedDocuments(
                query: .text(query), documents: documents.map { .text($0) })
            await MainActor.run {
                self.ranked = ranked
                self.memory = Self.formatMemory()
            }
        }
    }

    private func loadEmbedder(_ url: URL) async throws -> Qwen3VLEmbeddingSession {
        if let embedder { return embedder }
        let session = try await loadSession(url, task: .embedding)
        embedder = session
        return session
    }

    private func loadReranker(_ url: URL) async throws -> Qwen3VLEmbeddingSession {
        if let reranker { return reranker }
        let session = try await loadSession(url, task: .reranker)
        reranker = session
        return session
    }

    /// Load while holding security-scoped access to `url` (a no-op for in-container downloads).
    private func loadSession(_ url: URL, task: Qwen3VLTask) async throws -> Qwen3VLEmbeddingSession {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        return try await Qwen3VLEmbeddingSession.load(.init(modelDirectory: url, task: task))
    }

    // MARK: - Driving

    private func run(_ label: String, _ work: @escaping () async throws -> Void) {
        isBusy = true
        status = label
        Task {
            do {
                try await work()
                status = "Done"
            } catch {
                status = "Error: \(error)"
            }
            isBusy = false
        }
    }

    // MARK: - Images

    func addImages(_ urls: [URL]) {
        for url in urls {
            // Hold security-scoped access just long enough to decode the file.
            let scoped = url.startAccessingSecurityScopedResource()
            if let cg = Self.loadCGImage(url) {
                images.append((name: url.lastPathComponent, image: cg))
            }
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
    }

    func removeImage(at index: Int) {
        guard images.indices.contains(index) else { return }
        images.remove(at: index)
    }

    static func loadCGImage(_ url: URL) -> CGImage? {
        // Read the bytes eagerly (the caller holds security-scoped access) and decode from
        // in-memory data, then redraw into a *detached* bitmap. `CGImageSourceCreateWithURL`
        // produces a lazily memory-mapped image that re-opens the file on first pixel access —
        // which fails once the sandbox grant for a user-picked file has lapsed.
        guard let data = try? Data(contentsOf: url),
            let src = CGImageSourceCreateWithData(data as CFData, nil),
            let cg = CGImageSourceCreateImageAtIndex(
                src, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { return nil }
        guard
            let ctx = CGContext(
                data: nil, width: cg.width, height: cg.height, bitsPerComponent: 8,
                bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return cg }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
        return ctx.makeImage() ?? cg
    }

    // MARK: - Helpers

    private static func lines(_ s: String) -> [String] {
        s.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter {
            !$0.isEmpty
        }
    }

    private static func cosineMatrix(_ vectors: [[Float]]) -> [[Float]] {
        func cos(_ a: [Float], _ b: [Float]) -> Float {
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< min(a.count, b.count) {
                dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
            }
            let d = (na.squareRoot() * nb.squareRoot())
            return d > 0 ? dot / d : 0
        }
        return vectors.map { a in vectors.map { b in cos(a, b) } }
    }

    private static func formatMemory() -> String {
        let m = Qwen3VLEmbeddingSession.memorySnapshot()
        func mb(_ b: Int) -> String { String(format: "%.0f MB", Double(b) / 1_048_576) }
        return "active \(mb(m.active)) · cache \(mb(m.cache)) · peak \(mb(m.peak))"
    }
}

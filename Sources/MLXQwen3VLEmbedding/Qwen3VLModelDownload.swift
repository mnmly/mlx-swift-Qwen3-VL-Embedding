// Copyright © 2026
//
// Model download helper: fetch a Hugging Face repo snapshot to the local Hub cache so it can be
// loaded by path. Used by frontends that want an explicit "Download" action when the model isn't
// already present locally.

import Foundation
import Hub

extension Qwen3VLEmbeddingSession {
    /// Download a Hugging Face model snapshot and return the local directory.
    ///
    /// Downloads all files for `repoId` into the Hub cache (`HubApi`'s default download base) and
    /// returns the snapshot directory, suitable as ``Qwen3VLEmbeddingConfig/modelDirectory``.
    /// If the snapshot is already cached, this returns quickly without re-downloading.
    ///
    /// - Parameters:
    ///   - repoId: e.g. `"Qwen/Qwen3-VL-Embedding-2B"` or `"Qwen/Qwen3-VL-Reranker-2B"`.
    ///   - revision: the git revision to fetch (default `"main"`).
    ///   - progress: called with fractional progress in `0...1` as files download.
    @discardableResult
    public static func downloadModel(
        repoId: String,
        revision: String = "main",
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> URL {
        let hub = HubApi()
        return try await hub.snapshot(from: repoId, revision: revision) { p in
            progress(p.fractionCompleted)
        }
    }
}

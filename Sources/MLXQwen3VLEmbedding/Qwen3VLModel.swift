// Copyright © 2026
//
// Snapshot-directory resolution + processor-config patching for hand-picked
// Qwen3-VL checkpoints. Ported from mlx-swift-chandra's `ChandraModel`.

import Foundation

public enum Qwen3VLModel {

    /// Resolve a model directory, tolerating a Hugging Face cache repo root whose files
    /// live under `snapshots/<commit>/`.
    ///
    /// Resolution order:
    /// 1. If `url/config.json` exists, use `url` unchanged (a plain local snapshot).
    /// 2. If `url/snapshots/` exists, pick the `refs/main` commit if present, else the
    ///    newest snapshot containing `config.json`.
    /// 3. Otherwise return `url` unchanged (the loader surfaces a clear error).
    public static func resolveSnapshotDirectory(_ url: URL) -> URL {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.appendingPathComponent("config.json").path) { return url }

        let snapshots = url.appendingPathComponent("snapshots")
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: snapshots.path, isDirectory: &isDir), isDir.boolValue
        else { return url }

        func hasConfig(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        }

        if let ref = try? String(
            contentsOf: url.appendingPathComponent("refs/main"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !ref.isEmpty
        {
            let candidate = snapshots.appendingPathComponent(ref)
            if hasConfig(candidate) { return candidate }
        }

        let children =
            (try? fm.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        let valid = children.filter(hasConfig)
        let newest = valid.max { a, b in
            let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            return da < db
        }
        return newest ?? url
    }

    /// Ensure `preprocessor_config.json` carries a `processor_class` so the MLXVLM loader
    /// can pick the Qwen3-VL processor for a hand-picked snapshot. Idempotent; a no-op when
    /// the field is already present.
    public static func patchProcessorClass(in directory: URL) throws {
        let url = directory.appendingPathComponent("preprocessor_config.json")
        guard let data = try? Data(contentsOf: url),
            var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return }
        if obj["processor_class"] == nil {
            obj["processor_class"] = "Qwen3VLProcessor"
            let out = try JSONSerialization.data(
                withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
            try out.write(to: url)
        }
    }
}

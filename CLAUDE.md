# mlx-swift-qwen3vl-embedding

Qwen3-VL embedding + reranking on mlx-swift. The single library-side `Qwen3VLEmbeddingSession`
drives both the `qwen3vl-embed` CLI and the SwiftUI demo (shared-driver pattern) — keep model and
prompt logic in the library, never in a frontend.

The Qwen3-VL backbone lives in `mlx-swift-lm`. Only the *model* is vendored here
(`Sources/MLXQwen3VLEmbedding/Vendored/Qwen3VLBackbone.swift`) to expose `last_hidden_state`; the
processor and config are reused from `MLXVLM`. See `Vendored/VENDORED.md` before touching it, and
re-diff against upstream when bumping `mlx-swift-lm`.

Build/test with `xcodebuild` (not `swift test` — it can't load MLX's Metal lib). Parity is enforced
by `Tests/.../ParityTests.swift` against fixtures from `Tools/gen_fixtures.py`.

## Documentation

`MLXQwen3VLEmbedding` ships DocC-generated reference docs (see
`Sources/MLXQwen3VLEmbedding/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public symbols are published.**

When you add or modify a `public` declaration:

- Write a `///` doc comment: one-sentence summary, then a paragraph only if the *why* is
  non-obvious. Don't restate the signature.
- Document each parameter with `- Parameter name:` using the **internal** name when there's an
  external label (DocC warns otherwise).
- Cross-reference with double-backtick links, e.g. `` ``Qwen3VLEmbeddingSession/embed(texts:)`` ``.
  Link syntax is signature-sensitive: `foo(_:)` ≠ `foo(_:_:)`.
- File new top-level symbols under the right `## Topics` group in
  `Sources/MLXQwen3VLEmbedding/Documentation.docc/MLXQwen3VLEmbedding.md` (groups are organized by
  *user task*, not alphabetically).

Verify: `Scripts/build_docs.sh` exits 0 with no new "doesn't exist at" / "external name" warnings.
On this machine the default `swift` is a swiftly toolchain incompatible with the Xcode-beta SDK —
prefix the docs build with Xcode's toolchain on PATH (`export PATH="$(dirname "$(xcrun --find swift)"):$PATH"`).

# Vendored backbone

`Qwen3VLBackbone.swift` is a vendored copy of the Qwen3-VL **model** from upstream
`mlx-swift-lm`. It exists because the public `MLXVLM.Qwen3VL` exposes only **logits**
(`prepare()` → `.logits(LMOutput)`), and the embedder needs the pre-`lm_head`
`last_hidden_state`, which is produced inside an `internal` type.

## Provenance

- **Vendored from:** `ml-explore/mlx-swift-lm`, tag **`3.31.3`**,
  file `Libraries/MLXVLM/Models/Qwen3VL.swift`.
- **Consumed version:** this package now depends on `mlx-swift-lm` **`3.31.4`**
  (`from: "3.31.4"`) and `mlx-swift` `0.31.4`. The vendored *model* is the `3.31.3` copy; the
  processor, configuration, and message generator come from `3.31.4`'s `MLXVLM`.
- **Re-diffed `3.31.3 → 3.31.4` (2026-07):** the only change to `Qwen3VL.swift` in that bump is
  the M-RoPE delta cache moving from a `private var ropeDeltas` instance variable into a typed
  `LMOutput.State`/`Key` dict (threading a `state:` param through `callAsFunction`, returning
  `LMOutput(logits:, state:)`). This vendored copy keeps the `3.31.3` instance-var form.
  **It does not affect the embedding path:** `ropeDeltas` only caches across *generation* steps,
  and the embedder runs a single prefill (offset 0, `pixelValues != nil` resets it) — the cache
  is never read. Verified: compiles against `3.31.4` with exact image-embedding cosine parity.
  **No re-apply required.**
- Benign side effect of not tracking that refactor: `Qwen3VLBackbone.callAsFunction` keeps the
  old `(_ inputs: MLXArray, cache:)` signature (3.31.4 is
  `(_ input: LMInput.Text, cache:, state:) -> LMOutput`). The embedder never calls it (only
  `lastHiddenState`/`prepare`), so it is dead-but-harmless — it would only matter if this
  backbone were used to *generate*. (The `3.31.3 → 3.31.4` `Qwen3VLMessageGenerator.addToolMetadata`
  addition is in the message generator, which is **not** vendored — irrelevant.)

- **Ported to `3.32.3` (2026-10):** the package now requires `mlx-swift-lm` **`3.32.3`** /
  `mlx-swift` `0.32.3`. Only two API changes reach this file and the engine: the `LanguageModel`
  requirement `prepare(_:cache:windowSize:)` became `prepare(_:cache:state:prefill:)` (the
  vendored `prepare` now takes `state`/`prefill`, both unused — single prefill, no chunking), and
  `newCache(parameters:)` now `throws`. The vendored *model* is otherwise still the `3.31.3` copy.
  Upstream `3.32.3` also fixed the vision-MLP GELU (mod #5, now `gelu_pytorch_tanh` → `.tanh`) and
  reworked vision attention/M-RoPE for speed; none of that is pulled in here. Separately, the
  3.32 *processor* (not vendored) applies the sRGB tone curve to images itself, so
  `Qwen3VLPromptBuilder` no longer pre-applies it (doing both drops image cosine to ~0.945).

## What was vendored (and what was NOT)

Only the **model** is copied: `Qwen3VLVision`, `Qwen3VLLanguage`, and the top-level
backbone class. The **processor** (`Qwen3VLProcessor`), **configuration**
(`Qwen3VLConfiguration`), and **message generator** are *not* vendored — they are
reused from `MLXVLM` (imported), so there is exactly one source of truth for those.

## Local modifications vs upstream

1. Renamed the public model class `Qwen3VL` → `Qwen3VLBackbone` to avoid colliding
   with `MLXVLM.Qwen3VL` (both are linked).
2. Replaced the single `QwenVL.rotateHalf(...)` call (an `internal` MLXVLM helper) with
   the file-local, identical `Qwen3VLVision.rotateHalf(...)`, so MLXVLM's internal
   `QwenVL` is not needed.
3. Extracted the interleaved-M-RoPE position-id logic from `LanguageModel.callAsFunction`
   into a private `resolvePositionIds(...)` helper, shared by `callAsFunction` (logits)
   and the new `hiddenStates(...)` (pre-`lm_head`).
4. Extracted the vision/image-feature-merge prefill setup from `Qwen3VLBackbone.prepare`
   into a private `buildLanguageInputs(...)` helper, shared by `prepare` (logits) and
   the new **`lastHiddenState(_:cache:)`** — the reason this file exists.
5. **Bug fix (upstream):** the vision `MLP` activation was `GELU(approximation: .fast)`
   (`x·sigmoid(1.702x)`), but Qwen3-VL's vision config is `gelu_pytorch_tanh` (the tanh
   approximation = MLX `.precise`). The wrong approximation injected ~1% error per ViT block,
   compounding to image-embedding cosine ~0.93 vs the reference; `.precise` restores ~0.997.
   This should be reported upstream. (The `PatchMerger`'s `GELU()` = exact erf is correct —
   it matches PyTorch `nn.GELU()`.)

6. **Perf:** `Qwen3VLVision.Attention.callAsFunction` skips building the block-diagonal
   `cuSeqlens` mask when the call carries a single frame — the mask is then all-zero, i.e. a
   no-op, and building it costs an O(patches²) materialization plus a `cuSeqlens.asArray`
   GPU sync in *every* one of the 24 vision blocks. Numerically identical; measured 2.3×
   end-to-end on full-resolution images (1.35 → 3.17 img/s on an M5 Max) and it also drops
   peak GPU memory from 7.2 GB to 5.1 GB. The single-frame case is the only one the embedder
   and reranker ever hit. This should be reported upstream.
7. Widened `Qwen3VLBackbone.visionModel` and `.languageModel` from `private` to internal so
   `Qwen3VLBatchedForward.swift` (a pure addition outside this file, which keeps the upstream
   re-diff small) can drive them for the batched embedding path.

## Maintenance

When bumping `mlx-swift-lm`, re-diff `Qwen3VL.swift` against this file and re-apply
modifications 1–7. **Last re-diffed at `3.31.4`** — current, no re-apply needed (see the
re-diff note under *Provenance*). Both heads use this file: the reranker also reads the
pre-`lm_head` hidden state (it scores `(W[yes] − W[no]) · h` in float32, reading `W` from
`languageModel.lmHead` or, when tied, `languageModel.model.embedTokens`), so changes here affect
embedding *and* reranking.

### Retiring this file

This vendored copy exists for two reasons that upstream could remove — see the drafts under
[`upstream/`](./upstream/):

1. `01-vision-gelu-precise.md` — fixes the vision-MLP GELU (`.fast` → `.precise`); mod #5.
2. `02-expose-last-hidden-state.md` — adds a public pre-`lm_head` `lastHiddenState` accessor;
   the reason this file exists (mods 3–4).

If both land in a released `mlx-swift-lm`, this file can be deleted and both heads can run
against stock `MLXVLM.Qwen3VL` (the reranker additionally needs the `lm_head` / tied
`embed_tokens` weights, which the stock model exposes as `internal` today).

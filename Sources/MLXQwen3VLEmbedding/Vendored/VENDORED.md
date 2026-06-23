# Vendored backbone

`Qwen3VLBackbone.swift` is a vendored copy of the Qwen3-VL **model** from upstream
`mlx-swift-lm`. It exists because the public `MLXVLM.Qwen3VL` exposes only **logits**
(`prepare()` → `.logits(LMOutput)`), and the embedder needs the pre-`lm_head`
`last_hidden_state`, which is produced inside an `internal` type.

## Provenance

- **Upstream:** `ml-explore/mlx-swift-lm`, tag **`3.31.3`**,
  file `Libraries/MLXVLM/Models/Qwen3VL.swift`.
- This package depends on the same `mlx-swift-lm` `3.31.3` (and `mlx-swift` `0.31.4`) via
  remote SPM dependencies, so the vendored copy matches the consumed `MLXVLM` products.
  (At `3.31.3` the M-RoPE `ropeDeltas` is a `private var` instance variable; a later `main`
  commit moved it into a typed `LMOutput.State`/`Key` dict — this vendored copy uses the
  `3.31.3` instance-var form.)

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

## Maintenance

When bumping `mlx-swift-lm`, re-diff `Qwen3VL.swift` against this file and re-apply
modifications 1–4. The reranker path does **not** use this file (it uses stock
`MLXVLM.Qwen3VL`), so changes here only affect the embedder.

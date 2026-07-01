# [mlx-swift-lm] Qwen3-VL vision MLP uses `GELU(.fast)` instead of `gelu_pytorch_tanh` (`.precise`)

**Repo:** `ml-explore/mlx-swift-lm` · **File:** `Libraries/MLXVLM/Models/Qwen3VL.swift`
**Type:** bug fix (PR-ready, one line)

## Summary

Qwen3-VL's vision transformer `MLP` hardcodes the **sigmoid** GELU approximation:

```swift
// Qwen3VLVision.MLP.init
_activation.wrappedValue = GELU(approximation: .fast)   // x · sigmoid(1.702 x)
```

but Qwen3-VL's vision config specifies **`hidden_act: "gelu_pytorch_tanh"`**, which is the
**tanh** approximation — MLX's `GELU(approximation: .precise)`. These are numerically
different activations, so the wrong one is used on every vision-tower block.

## Impact

The error is small per block (~1%) but **compounds across the ViT depth**. Measured against
the HF `transformers` reference on the same image:

| vision MLP activation | image-embedding cosine vs reference |
|---|---|
| `.fast` (current) | ~0.93 |
| `.precise` (tanh, correct) | ~0.997 |

Anything that consumes the vision features is affected — most visibly representation /
embedding use, and to a lesser degree VQA logits.

## Location

`Libraries/MLXVLM/Models/Qwen3VL.swift`, `enum Qwen3VLVision` → `final class MLP`, in `init`:

```swift
_activation.wrappedValue = GELU(approximation: .fast)
```

(Note: the sibling `PatchMerger` uses `GELU()` = exact erf, which **is** correct — it matches
PyTorch `nn.GELU()`. Only the per-block `MLP` activation is wrong.)

## Minimal fix

```diff
-            _activation.wrappedValue = GELU(approximation: .fast)
+            // Qwen3-VL vision config: hidden_act = "gelu_pytorch_tanh" (tanh approx = MLX .precise)
+            _activation.wrappedValue = GELU(approximation: .precise)
```

## Preferred fix (respect the config)

The vision configuration already surfaces `hidden_act` (defaulting to `"gelu"`). Selecting the
approximation from it avoids silently baking in the wrong activation for other checkpoints:

```swift
static func gelu(for hiddenAct: String) -> GELU {
    switch hiddenAct {
    case "gelu_pytorch_tanh", "gelu_new": return GELU(approximation: .precise) // tanh
    case "gelu_fast", "quick_gelu":       return GELU(approximation: .fast)    // sigmoid
    default:                              return GELU()                        // exact erf
    }
}
```

Happy to open the PR with either form (minimal or config-driven).

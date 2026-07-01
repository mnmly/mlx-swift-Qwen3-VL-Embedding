# [mlx-swift-lm] Expose Qwen3-VL pre-`lm_head` `last_hidden_state` (for embedding heads)

**Repo:** `ml-explore/mlx-swift-lm` · **File:** `Libraries/MLXVLM/Models/Qwen3VL.swift`
**Type:** feature request (PR-ready)

## Motivation

Building a Qwen3-VL **embedding** head (the reference `Qwen3-VL-Embedding` pools the
pre-`lm_head` `last_hidden_state` at the last / EOS token) is currently **impossible against
the public API**: `Qwen3VL.prepare(...)` and `callAsFunction(...)` return only `.logits`, and
the trunk output (`norm(hidden)`, before `lm_head`) is produced inside the `internal`
`LanguageModel` / `Model` and never surfaced.

The result is that every embedding / representation consumer has to **vendor the entire
`Qwen3VL.swift` model file** just to add a single accessor — and then re-diff it on every
`mlx-swift-lm` bump. Exposing the hidden state upstream removes that maintenance burden for a
whole class of downstream uses (embeddings, rerankers that pool hidden states, probing, etc.).

## Current state

```swift
// Qwen3VL.prepare(...)
let languageOutput = languageModel(inputIds, cache: typedCache, state: nil,
    inputEmbeddings: inputEmbeddings, mask: nil, positionIds: nil,
    visualMask: visualMask, deepstackEmbeds: deepstackEmbeds,
    pixelValues: pixelValues, imageGridTHW: imageFrames, videoGridTHW: videoFrames)
return .logits(languageOutput)   // <- only path out
```

`languageModel` runs the trunk `model(...)` (already `norm`'d) and then applies `lmHead`.
There is no way to obtain the trunk output.

## Proposal

Add a public method that mirrors the existing prefill (vision merge + M-RoPE) but returns the
pre-`lm_head` hidden state instead of logits:

```swift
// On LanguageModel — mirror callAsFunction, return the trunk instead of lmHead(trunk):
func hiddenStates(_ inputIds: MLXArray?, cache: [KVCache]?, state: LMOutput.State?,
                  inputEmbeddings: MLXArray?, mask: MLXArray?, positionIds: MLXArray?,
                  visualMask: MLXArray?, deepstackEmbeds: [MLXArray]?,
                  pixelValues: MLXArray?, imageGridTHW: [THW]?, videoGridTHW: [THW]?) -> MLXArray {
    let positionIds = resolvePositionIds(/* … */)          // extract from callAsFunction
    return model(inputIds, cache: cache, /* … */, positionIds: positionIds, /* … */)  // no lmHead
}

// On Qwen3VL (the VLMModel) — mirror prepare, return the trunk hidden state:
public func lastHiddenState(_ input: LMInput, cache: [any KVCache]) throws -> MLXArray {
    let li = try buildLanguageInputs(input)               // extract from prepare
    return languageModel.hiddenStates(li.inputIds, cache: castCache(cache), state: nil, /* … */)
}
```

This needs two small refactors that are purely mechanical (no behavior change):

1. Extract the M-RoPE position-id resolution out of `LanguageModel.callAsFunction` into a
   private `resolvePositionIds(...)`, shared by `callAsFunction` (logits) and `hiddenStates`.
2. Extract the vision / image-feature-merge prefill out of `Qwen3VL.prepare` into a private
   `buildLanguageInputs(...)`, shared by `prepare` (logits) and `lastHiddenState`.

### Lighter alternative

If you'd rather not add the method, making `languageModel` / the trunk `model` (or a
`hiddenStates` accessor) `public` would also unblock consumers — but the value here is sharing
the *prefill* (vision merge + M-RoPE), so a `lastHiddenState` on the model reads cleaner.

## Reference implementation

We already run exactly this as a vendored copy (`Qwen3VLBackbone.swift`) — the two extractions
above plus `hiddenStates(...)` / `lastHiddenState(_:cache:)`. Glad to port it into a PR against
`Qwen3VL.swift` if the API shape is agreeable.

(Filed alongside a separate one-line vision-GELU correctness fix — see `01-vision-gelu-precise.md`.)

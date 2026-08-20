// Copyright © 2026
//
// Batched forward pass for the embedding head.
//
// Why this exists: the stock path embeds one item per model call, so corpus-scale image
// indexing runs the decoder's prefill at batch 1 and pays the per-call CPU work — prompt
// assembly, the host-side M-RoPE index scan, a GPU sync per image — once per image. Folding
// B prompts into one graph amortizes both.
//
// What is and is not batched, decided by measurement rather than symmetry:
//
//   • Language model — batched. Prompts are right-padded to the batch maximum. Attention is
//     causal, so a real token at position i never sees a pad at j > i and no extra masking
//     is needed; pooling reads each row's own `length - 1` position instead of the last
//     column. Worth 1.7× on a small-image corpus.
//   • Vision tower — *not* batched, one image per call. Qwen3-VL is a native-resolution
//     model, so images in a batch rarely share a patch grid; batching them means padding to
//     the batch maximum, and vision attention is quadratic in the patch count, so padding a
//     900-patch image out to a 5000-patch neighbour costs more than batching saves. A
//     grid-grouped variant that padded nothing was also built and measured: it came out even
//     with the per-image loop on small images and 5–10% *behind* it on large ones, because a
//     single native-resolution image (up to 5120 patches) already fills the GPU on its own.
//     The per-image loop is what shipped.
//
// This file is a pure addition; it lives outside `Vendored/Qwen3VLBackbone.swift` so the
// upstream re-diff stays small (that file only widens two access modifiers for it).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Backbone

extension Qwen3VLBackbone {

    /// One image's patch rows plus the grid they came from.
    struct VisionRow {
        var grid: THW
        var pixels: MLXArray
    }

    /// One right-padded batch of prompts, at most one image each.
    struct BatchedInput {
        /// `[B, seq]` prompt tokens, right-padded with a non-image filler token.
        var tokens: MLXArray
        /// `[3, B, seq]` interleaved-M-RoPE position ids, one row per prompt.
        var positionIds: MLXArray
        /// Patch rows for the image-bearing prompts, in row order.
        var visionRows: [VisionRow]
    }

    /// ``lastHiddenState(_:cache:)`` for a whole batch: `[B, seq, hidden]`.
    ///
    /// No KV cache is allocated — this is a single prefill and nothing reads the cache
    /// afterwards, which at batch 16 saves gigabytes the unbatched path spends on keys and
    /// values it never uses.
    func lastHiddenStateBatch(_ input: BatchedInput) -> MLXArray {
        let textEmbeds = languageModel.model.embedTokens(input.tokens)

        var inputEmbeddings = textEmbeds
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]?

        if !input.visionRows.isEmpty {
            let (visionHidden, deepstackOutputs) = visionFeatures(input.visionRows)

            let (merged, mask) = Self.scatterImageFeatures(
                imageFeatures: visionHidden.asType(textEmbeds.dtype),
                inputEmbeds: textEmbeds,
                inputIds: input.tokens,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex)
            inputEmbeddings = merged
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { $0.asType(textEmbeds.dtype) }
            }
        }

        return languageModel.hiddenStates(
            input.tokens,
            cache: nil,
            inputEmbeddings: inputEmbeddings,
            mask: nil,
            positionIds: input.positionIds,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil)
    }

    /// Run each image through the vision tower on its own and concatenate the features in
    /// row order — the layout the language-side scatter expects. Nothing is evaluated here;
    /// the calls join one lazy graph the caller evaluates once for the whole batch.
    private func visionFeatures(_ rows: [VisionRow]) -> (MLXArray, [MLXArray]) {
        let dtype = visionModel.patchEmbed.proj.weight.dtype
        var mains: [MLXArray] = []
        var deepstacks: [[MLXArray]] = []
        mains.reserveCapacity(rows.count)
        for row in rows {
            let (main, deepstack) = visionModel(row.pixels.asType(dtype), gridTHW: [row.grid])
            mains.append(main)
            deepstacks.append(deepstack)
        }
        let levels = deepstacks.first?.count ?? 0
        guard rows.count > 1 else { return (mains[0], deepstacks[0]) }
        return (
            concatenated(mains, axis: 0),
            (0 ..< levels).map { level in concatenated(deepstacks.map { $0[level] }, axis: 0) }
        )
    }

    /// Batch-safe twin of the backbone's `mergeInputIdsWithImageFeatures`: scatter one
    /// feature row into each image-token slot, on-device.
    ///
    /// Row-major flattening of `[B, seq]` visits row 0's image tokens before row 1's, which is
    /// exactly the order the vision loop emits features in, so a running count of image tokens
    /// indexes them directly. Unlike the unbatched version this never calls `.item()`, so it
    /// adds no GPU sync per batch.
    private static func scatterImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) -> (MLXArray, MLXArray) {
        let mask2D =
            (inputIds .== MLXArray(imageTokenIndex)) .|| (inputIds .== MLXArray(videoTokenIndex))
        let visualMask = mask2D.asType(.bool)
        guard imageFeatures.dim(0) > 0 else { return (inputEmbeds, visualMask) }

        let b = inputEmbeds.dim(0), s = inputEmbeds.dim(1), h = inputEmbeds.dim(2)
        let flatMask = mask2D.asType(.int32).reshaped([-1])  // [B*seq]
        let segment = maximum(cumsum(flatMask, axis: 0) - 1, MLXArray(Int32(0)))
        let gathered = imageFeatures[segment].reshaped([b, s, h])
        let maskF = mask2D.asType(inputEmbeds.dtype).reshaped([b, s, 1])
        return (inputEmbeds * (1 - maskF) + gathered * maskF, visualMask)
    }
}

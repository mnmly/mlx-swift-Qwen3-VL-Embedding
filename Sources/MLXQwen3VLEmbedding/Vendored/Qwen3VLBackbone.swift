// Vendored from ml-explore/mlx-swift-lm @ e48c6b0 (tag 3.31.3-68),
//   Libraries/MLXVLM/Models/Qwen3VL.swift — see VENDORED.md.
//
// Only the *model* is vendored (Vision + Language + the backbone class). The
// processor, configuration, and message generator are reused from MLXVLM, so this
// file deliberately omits them. Local modifications vs upstream:
//   - Renamed `Qwen3VL` -> `Qwen3VLBackbone` (avoids colliding with MLXVLM.Qwen3VL).
//   - Replaced the one `QwenVL.rotateHalf` call with the file-local
//     `Qwen3VLVision.rotateHalf` (identical) so MLXVLM's internal QwenVL isn't needed.
//   - Added `lastHiddenState(_:cache:)` exposing the pre-`lm_head` hidden state,
//     which the public `Qwen3VL`/`VLMModel` surface does not expose (logits only).
//   - Widened `Qwen3VLBackbone.visionModel` and `.languageModel` from private to internal
//     so the batched forward (`Qwen3VLBatchedForward.swift`, a pure addition living outside
//     this file to keep the upstream diff small) can drive them. Bodies are untouched.
//   - `Qwen3VLVision.Attention.callAsFunction` skips the block-diagonal mask when the
//     batch holds a single sequence (the mask is then all-zero, i.e. a no-op) — this
//     also removes a per-block `cuSeqlens.asArray` GPU sync. Numerically identical.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM

private enum Qwen3VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

// MARK: - Vision

enum Qwen3VLVision {

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq =
                1.0
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                        / Float(dimension)
                )
            let seq = MLXArray(0 ..< sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize
            ).movedAxis(source: 1, destination: 4)

            states = proj(states)
            states = states.reshaped(-1, hiddenSize)
            return states
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        let usePostShuffleNorm: Bool

        @ModuleInfo(key: "norm") var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(config: Qwen3VLConfiguration.VisionConfiguration, usePostShuffleNorm: Bool) {
            self.hiddenSize =
                config.hiddenSize * (config.spatialMergeSize * config.spatialMergeSize)
            self.usePostShuffleNorm = usePostShuffleNorm

            let normDim = usePostShuffleNorm ? hiddenSize : config.hiddenSize
            _norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize)
            _linear2.wrappedValue = Linear(hiddenSize, config.outHiddenSize)
            _activation.wrappedValue = GELU()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x
            if usePostShuffleNorm {
                states = states.reshaped(-1, hiddenSize)
            }
            states = norm(states)
            states = states.reshaped(-1, hiddenSize)
            states = linear1(states)
            states = activation(states)
            states = linear2(states)
            return states
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            var qkvStates = qkv(x)
            qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, headDim)
            qkvStates = qkvStates.transposed(1, 0, 2, 3)

            let parts = split(qkvStates, parts: 3, axis: 0)
            var queries = parts[0][0, 0..., 0..., 0...]
            var keys = parts[1][0, 0..., 0..., 0...]
            var values = parts[2][0, 0..., 0..., 0...]

            queries = applyRotary(queries, freqs: rotaryPosEmb)
            keys = applyRotary(keys, freqs: rotaryPosEmb)

            queries = queries.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

            // `cuSeqlens` delimits the per-frame blocks of the block-diagonal mask. Its
            // *shape* (frames + 1) is known without evaluating it, so a single frame can
            // take the maskless path: the mask would be all-zero anyway, and building it
            // costs an O(seq²) materialization plus a `cuSeqlens.asArray` GPU sync in every
            // one of the 24 vision blocks. (Local modification — see the file header.)
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if cuSeqlens.dim(0) <= 2 {
                maskMode = .none
            } else {
                var mask = ones([1, sequenceLength, sequenceLength], dtype: queries.dtype)
                mask = mask * MLXArray(-1e9, dtype: queries.dtype)

                let seqlens = cuSeqlens.asArray(Int.self)
                for idx in 1 ..< seqlens.count {
                    let start = seqlens[idx - 1]
                    let end = seqlens[idx]
                    mask[0..., start ..< end, start ..< end] = MLXArray(0, dtype: queries.dtype)
                }
                maskMode = .array(mask)
            }

            let attended = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: maskMode
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(attended)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(dim: Int, hiddenDim: Int) {
            _linear1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _linear2.wrappedValue = Linear(hiddenDim, dim, bias: true)
            // Qwen3-VL vision config uses `gelu_pytorch_tanh` (the tanh approximation = MLX
            // `.precise`). Upstream Qwen3VL.swift used `.fast` (x·sigmoid(1.702x)), a different
            // approximation that injects ~1% error per block (vision parity fix).
            _activation.wrappedValue = GELU(approximation: .precise)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: LayerNorm
        @ModuleInfo(key: "norm2") var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            var states = hiddenStates
            states =
                states + attention(norm1(states), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb)
            states = states + mlp(norm2(states))
            return states
        }
    }

    final class VisionModel: Module {

        let config: Qwen3VLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let numGridPerSide: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]
        let deepstackVisualIndexes: [Int]

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            self.config = config
            self.spatialMergeSize = config.spatialMergeSize
            self.numGridPerSide = Int(sqrt(Double(config.numPositionEmbeddings)))
            self.deepstackVisualIndexes = config.deepstackVisualIndexes

            _patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDim = config.hiddenSize / config.numHeads
            _rotaryEmbedding.wrappedValue = VisionRotaryEmbedding(dimension: headDim / 2)

            _posEmbed.wrappedValue = Embedding(
                embeddingCount: config.numPositionEmbeddings,
                dimensions: config.hiddenSize)

            _blocks.wrappedValue = (0 ..< config.depth).map { _ in VisionBlock(config) }
            _merger.wrappedValue = PatchMerger(config: config, usePostShuffleNorm: false)

            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config: config, usePostShuffleNorm: true)
            }
        }

        private func rotaryPositionEmbedding(_ grids: [THW]) -> MLXArray {
            guard let maxHW = grids.map({ max($0.h, $0.w) }).max(), maxHW > 0 else {
                return MLXArray.zeros([1, 1], dtype: .float32)
            }

            let freqTable = rotaryEmbedding(sequenceLength: maxHW)
            let halfDim = freqTable.dim(-1)

            let merge = spatialMergeSize
            var allCoords: [MLXArray] = []
            let mergeScalar = MLXArray(Int32(merge))

            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge

                guard mergedH > 0, mergedW > 0 else { continue }

                // Generate block and intra-block indices fully in MLX
                var blockRows = MLXArray(0 ..< mergedH).asType(.int32)
                blockRows = blockRows.reshaped([mergedH, 1, 1, 1])

                var blockCols = MLXArray(0 ..< mergedW).asType(.int32)
                blockCols = blockCols.reshaped([1, mergedW, 1, 1])

                let intra = MLXArray(0 ..< merge).asType(.int32)
                let intraRow = intra.reshaped([1, 1, merge, 1])
                let intraCol = intra.reshaped([1, 1, 1, merge])

                // Broadcast arithmetic mirrors the Python implementation
                var hIndex = blockRows * mergeScalar + intraRow
                var wIndex = blockCols * mergeScalar + intraCol

                hIndex = broadcast(hIndex, to: [mergedH, mergedW, merge, merge])
                wIndex = broadcast(wIndex, to: [mergedH, mergedW, merge, merge])

                // Flatten and stack coordinate pairs
                let hFlattened = hIndex.flattened()
                let wFlattened = wIndex.flattened()
                var coords = stacked([hFlattened, wFlattened], axis: -1)

                // Repeat for temporal frames
                if grid.t > 1 {
                    coords = tiled(coords, repetitions: [grid.t, 1])
                }

                allCoords.append(coords)
            }

            guard !allCoords.isEmpty else {
                return MLXArray.zeros([0, halfDim * 2], dtype: freqTable.dtype)
            }

            // Concatenate all coordinate pairs
            let allCoordsConcat = concatenated(allCoords, axis: 0)  // (total_tokens, 2)

            // Extract h and w indices and lookup embeddings
            let hIndices = allCoordsConcat[0..., 0].asType(.int32)
            let wIndices = allCoordsConcat[0..., 1].asType(.int32)

            let hEmbeds = freqTable[hIndices, 0...]
            let wEmbeds = freqTable[wIndices, 0...]

            // Concatenate height and width embeddings
            return concatenated([hEmbeds, wEmbeds], axis: -1)
        }

        private func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
            let hiddenSize = config.hiddenSize
            let maxIndex = numGridPerSide - 1

            // Step 1: Collect all indices and weights from all grids using MLX ops
            var cornerIndices: [[MLXArray]] = Array(repeating: [], count: 4)
            var cornerWeights: [[MLXArray]] = Array(repeating: [], count: 4)
            var gridSizes: [Int] = []

            for grid in grids {
                let h = grid.h
                let w = grid.w
                gridSizes.append(h * w)

                // Create linspace indices using broadcasting
                var hLinspace = MLXArray(0 ..< h).asType(.float32)
                hLinspace = hLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, h - 1)))

                var wLinspace = MLXArray(0 ..< w).asType(.float32)
                wLinspace = wLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, w - 1)))

                // Get floor/ceil and deltas
                let hFloor = hLinspace.asType(.int32)
                let hCeil = minimum(hFloor + 1, maxIndex)
                let dh = hLinspace - hFloor.asType(.float32)

                let wFloor = wLinspace.asType(.int32)
                let wCeil = minimum(wFloor + 1, maxIndex)
                let dw = wLinspace - wFloor.asType(.float32)

                // Broadcast to create meshgrid
                let hFloorExpanded = expandedDimensions(hFloor, axis: 1)  // (h, 1)
                let hCeilExpanded = expandedDimensions(hCeil, axis: 1)
                let wFloorExpanded = expandedDimensions(wFloor, axis: 0)  // (1, w)
                let wCeilExpanded = expandedDimensions(wCeil, axis: 0)

                let baseH = hFloorExpanded * numGridPerSide
                let baseHCeil = hCeilExpanded * numGridPerSide

                // Compute 4 corner indices
                cornerIndices[0].append((baseH + wFloorExpanded).flattened())
                cornerIndices[1].append((baseH + wCeilExpanded).flattened())
                cornerIndices[2].append((baseHCeil + wFloorExpanded).flattened())
                cornerIndices[3].append((baseHCeil + wCeilExpanded).flattened())

                // Compute bilinear weights
                let dhExpanded = expandedDimensions(dh, axis: 1)
                let dwExpanded = expandedDimensions(dw, axis: 0)

                cornerWeights[0].append(((1 - dhExpanded) * (1 - dwExpanded)).flattened())
                cornerWeights[1].append(((1 - dhExpanded) * dwExpanded).flattened())
                cornerWeights[2].append((dhExpanded * (1 - dwExpanded)).flattened())
                cornerWeights[3].append((dhExpanded * dwExpanded).flattened())
            }

            guard !cornerIndices[0].isEmpty else {
                return MLXArray.zeros([0, hiddenSize], dtype: posEmbed.weight.dtype)
            }

            // Step 2: Batch embedding lookup
            let indicesTensors = cornerIndices.map { concatenated($0, axis: 0).asType(.int32) }
            let weightsTensors = cornerWeights.map {
                concatenated($0, axis: 0).asType(posEmbed.weight.dtype)
            }

            let totalPatches = indicesTensors[0].dim(0)
            var patchPosEmbeds = MLXArray.zeros(
                [totalPatches, hiddenSize], dtype: posEmbed.weight.dtype)

            for cornerIdx in 0 ..< 4 {
                let cornerEmbeds = posEmbed(indicesTensors[cornerIdx])
                let weighted =
                    cornerEmbeds * expandedDimensions(weightsTensors[cornerIdx], axis: -1)
                patchPosEmbeds = patchPosEmbeds + weighted
            }

            // Step 3: Split by grid (like Python lines 344-349)
            var patchPosEmbedsSplit: [MLXArray] = []
            var offset = 0

            for size in gridSizes {
                let slice = patchPosEmbeds[offset ..< (offset + size), 0...]
                patchPosEmbedsSplit.append(slice)
                offset += size
            }

            // Step 4: Process each grid (like Python lines 354-371)
            var resultEmbeds: [MLXArray] = []
            let merge = spatialMergeSize

            for (gridIdx, grid) in grids.enumerated() {
                let posEmbed = patchPosEmbedsSplit[gridIdx]
                let h = grid.h
                let w = grid.w
                let t = grid.t

                let featureDim = posEmbed.dim(-1)

                // Repeat for temporal dimension
                var temporalEmbeds = tiled(posEmbed, repetitions: [t, 1])

                // Reshape for merge pattern
                temporalEmbeds = temporalEmbeds.reshaped(t, h, w, featureDim)
                temporalEmbeds = temporalEmbeds.reshaped(
                    t,
                    h / merge,
                    merge,
                    w / merge,
                    merge,
                    featureDim
                )
                temporalEmbeds = temporalEmbeds.transposed(0, 1, 3, 2, 4, 5)
                temporalEmbeds = temporalEmbeds.reshaped(-1, featureDim)

                resultEmbeds.append(temporalEmbeds)
            }

            return concatenated(resultEmbeds, axis: 0)
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> MLXArray {
            var seqLengths: [MLXArray] = []

            for grid in grids {
                let perFrame = grid.h * grid.w
                let repeated = tiled(MLXArray(perFrame), repetitions: [grid.t])
                seqLengths.append(repeated)
            }

            guard !seqLengths.isEmpty else {
                return MLXArray(0, dtype: .int32)
            }

            let concatSeqLengths = concatenated(seqLengths).asType(.int32)

            let cumSum = concatSeqLengths.cumsum()

            return padded(
                cumSum, widths: [IntOrPair((1, 0))], mode: .constant,
                value: MLXArray(0, dtype: cumSum.dtype))
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) -> (MLXArray, [MLXArray]) {
            var hiddenStates = patchEmbed(pixelValues)

            let posEmbeds = positionalEmbeddings(gridTHW)
            hiddenStates = hiddenStates + posEmbeds

            let rotaryEmbeds = rotaryPositionEmbedding(gridTHW)
            let cuSeqlens = cumulativeSequenceLengths(gridTHW)

            var deepstackOutputs: [MLXArray] = []

            for (index, block) in blocks.enumerated() {
                hiddenStates = block(hiddenStates, cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryEmbeds)
                if let dsIndex = deepstackVisualIndexes.firstIndex(of: index) {
                    let feature = deepstackMergers[dsIndex](hiddenStates)
                    deepstackOutputs.append(feature)
                }
            }

            hiddenStates = merger(hiddenStates)
            return (hiddenStates, deepstackOutputs)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                } else if key.contains("patch_embed.proj.weight") {
                    if value.ndim == 5 && value.dim(-1) == config.inChannels {
                        sanitized[key] = value
                    } else {
                        sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Language

enum Qwen3VLLanguage {

    final class RotaryEmbedding {

        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(headDim: Int, base: Double, ropeScaling: Qwen3VLConfiguration.RoPEScaling?) {
            var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            freq = freq / Float(headDim)
            let baseArray = MLXArray(Float(base))
            self.invFreq = 1.0 / pow(baseArray, freq)
            self.mropeSection = ropeScaling?.mropeSection ?? [24, 20, 20]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqs_t = freqs[0, 0..., 0..., 0...]  // (bs, seq_len, head_dim // 2)

            let dims = freqs_t.dim(-1)
            var slices: [MLXArray] = []

            for idx in 0 ..< dims {
                var slice = freqs_t[0..., 0..., idx]

                for (dimIndex, offset) in [(1, 1), (2, 2)] {
                    let end = min(mropeSection[dimIndex] * 3, dims)
                    if idx >= offset && idx < end && (idx - offset) % 3 == 0 {
                        slice = freqs[dimIndex, 0..., 0..., idx]
                        break
                    }
                }

                slices.append(slice)
            }

            return stacked(slices, axis: -1)
        }

        func callAsFunction(positionIds: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = positionIds[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds, repetitions: [3, 1, 1])
            }

            let pos = positionIds.asType(.float32)
            var invFreq = self.invFreq.asType(.float32)
            invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            let cosValues = cos(emb).asType(dtype)
            let sinValues = sin(emb).asType(dtype)
            return (cosValues, sinValues)
        }
    }

    static func applyMultimodalRotary(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var cos = cos
        var sin = sin
        cos = expandedDimensions(cos, axis: 1)
        sin = expandedDimensions(sin, axis: 1)
        let qEmbedded = (q * cos) + (Qwen3VLVision.rotateHalf(q) * sin)
        let kEmbedded = (k * cos) + (Qwen3VLVision.rotateHalf(k) * sin)
        return (qEmbedded, kEmbedded)
    }

    final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else {
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)

            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            let result = wo(output)

            return result
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var residual = attention(
                inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }

    final class Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIds: positionIds)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let visualMask
                {

                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualMask: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            // Add `visualEmbeds` (one row per visual token, in order) to the hidden
            // states at the visual positions marked by `visualMask` ([B*S] bool),
            // fully on-device: per-position feature index = cumsum(mask)-1, gather
            // the embed row per position, keep it only where the mask is set.
            // Replaces a host round-trip (asArray(Bool) + Swift index loop).
            guard visualEmbeds.dim(0) > 0 else { return hiddenStates }
            let b = hiddenStates.dim(0), s = hiddenStates.dim(1), h = hiddenStates.dim(2)
            let flat = visualMask.asType(.int32).reshaped([-1])
            let seg = maximum(cumsum(flat, axis: 0) - 1, MLXArray(Int32(0)))
            let gathered = visualEmbeds[seg].reshaped([b, s, h])
            let maskF = visualMask.asType(hiddenStates.dtype).reshaped([b, s, 1])
            return hiddenStates + gathered * maskF
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3VLConfiguration
        let textConfig: Qwen3VLConfiguration.TextConfiguration
        var kvHeads: [Int]

        // mlx-swift-lm 3.31.3 carries M-RoPE deltas as an instance variable (post-3.31.3 moved
        // this into the typed LMOutput.State dict). For single-prefill embedding/reranking it is
        // computed once and not reused.
        private var ropeDeltas: MLXArray? = nil

        init(_ config: Qwen3VLConfiguration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.numKeyValueHeads,
                count: config.textConfiguration.numHiddenLayers)

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabSize,
                    bias: false)
            }
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> LMOutput {
            let positionIds = resolvePositionIds(
                inputIds: inputIds, cache: cache,
                inputEmbeddings: inputEmbeddings, mask: mask,
                providedPositionIds: providedPositionIds, pixelValues: pixelValues,
                imageGridTHW: imageGridTHW, videoGridTHW: videoGridTHW)

            var output = model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIds: positionIds,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds)

            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }

            return LMOutput(logits: output)
        }

        /// Pre-`lm_head` hidden state (= `norm(hidden)`, i.e. `last_hidden_state`).
        /// Mirrors ``callAsFunction`` without the vocabulary projection — added for the
        /// embedding head (local vendoring addition, see VENDORED.md).
        func hiddenStates(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> MLXArray {
            let positionIds = resolvePositionIds(
                inputIds: inputIds, cache: cache,
                inputEmbeddings: inputEmbeddings, mask: mask,
                providedPositionIds: providedPositionIds, pixelValues: pixelValues,
                imageGridTHW: imageGridTHW, videoGridTHW: videoGridTHW)

            return model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIds: positionIds,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds)
        }

        /// Shared interleaved-M-RoPE position-id resolution for ``callAsFunction`` and
        /// ``hiddenStates`` (extracted verbatim from the upstream `callAsFunction` body).
        private func resolvePositionIds(
            inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            providedPositionIds: MLXArray?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> MLXArray? {
            if pixelValues != nil {
                ropeDeltas = nil
            }

            var positionIds = providedPositionIds

            if positionIds == nil && (mask == nil || mask?.ndim == 2) {
                if (cache?.first?.offset ?? 0) == 0 || ropeDeltas == nil || cache == nil {
                    if let inputIds {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputIds,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenIndex,
                            videoTokenId: config.videoTokenIndex,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: mask)

                        positionIds = computed
                        ropeDeltas = deltas
                    } else if let cache, ropeDeltas == nil {
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0

                        var base = MLXArray(0 ..< seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        let offsetValue = MLXArray(currentOffset).asType(.int32)
                        base = base + offsetValue

                        positionIds = base[.newAxis, 0..., 0...]
                        positionIds = tiled(positionIds!, repetitions: [3, batch, seqLength])
                    }
                } else if let cache, let ropeDeltas {
                    let batch = (inputIds ?? inputEmbeddings!).dim(0)
                    let seqLength = (inputIds ?? inputEmbeddings!).dim(1)

                    let lastCacheOffset = cache.last?.offset ?? 0

                    var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = base[.newAxis, 0...]
                    base = broadcast(base, to: [batch, seqLength])

                    if delta.dim(0) == 1 && batch > 1 {
                        delta = repeated(delta, count: batch, axis: 0)
                    }

                    base = base + delta

                    positionIds = base[.newAxis, 0..., 0...]
                    positionIds = broadcast(positionIds!, to: [3, batch, seqLength])
                }
            }

            return positionIds
        }

    }
}

extension Qwen3VLLanguage {

    static func getRopeIndex(
        inputIds: MLXArray,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?,
        spatialMergeSize: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {

        let (batchSize, seqLength) = (inputIds.dim(0), inputIds.dim(1))

        var positionIds = MLXArray(0 ..< seqLength).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0...], to: [batchSize, seqLength])

        guard inputIds.ndim > 0, imageGridTHW != nil || videoGridTHW != nil else {
            let positionIds3D = broadcast(
                positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
            let zeros = MLXArray.zeros([batchSize], dtype: .int32)
            return (positionIds3D, zeros)
        }

        positionIds = ones(like: inputIds).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])

        var mropePositionDeltas: [Int] = []
        let mask = attentionMask ?? ones(like: inputIds)

        // Process each batch item (assume batch=1 for now)
        for batchIdx in 0 ..< batchSize {
            var batchInputIds = inputIds[batchIdx, 0...]

            // Mask out padding - use where from MLX module
            batchInputIds = `where`(
                mask[batchIdx, 0...] .== 1, batchInputIds, zeros(like: batchInputIds))

            // Count images and videos in this sequence
            let visionStartMask = (batchInputIds .== MLXArray(visionStartTokenId))
            let visionStartWeighted = `where`(
                visionStartMask, MLXArray(0 ..< seqLength), zeros(like: batchInputIds))
            let visionStartIdx = argMax(visionStartWeighted).item(Int.self)

            guard visionStartIdx < seqLength - 1 else {
                continue  // No vision tokens
            }

            let imageNums = ((batchInputIds .== MLXArray(imageTokenId)).asType(.int32).sum()).item(
                Int.self)
            let videoNums = ((batchInputIds .== MLXArray(videoTokenId)).asType(.int32).sum()).item(
                Int.self)

            let inputTokens = batchInputIds.asArray(Int32.self).map { Int($0) }
            var llmPosIdsList: [MLXArray] = []

            var st = 0
            var remainImages = imageNums
            var remainVideos = videoNums
            var imageIndex = 0
            var videoIndex = 0

            // Process each image/video in sequence
            for _ in 0 ..< (imageNums + videoNums) {
                // Find next image/video token position
                let edImage: Int
                if remainImages > 0, let idx = inputTokens[st...].firstIndex(of: imageTokenId) {
                    edImage = idx
                } else {
                    edImage = inputTokens.count + 1
                }

                let edVideo: Int
                if remainVideos > 0, let idx = inputTokens[st...].firstIndex(of: videoTokenId) {
                    edVideo = idx
                } else {
                    edVideo = inputTokens.count + 1
                }

                let (t, h, w, ed): (Int, Int, Int, Int)
                if edImage < edVideo {
                    // Process image
                    guard let grid = imageGridTHW, imageIndex < grid.count else { break }
                    (t, h, w) = grid[imageIndex].values
                    imageIndex += 1
                    remainImages -= 1
                    ed = edImage
                } else {
                    // Process video
                    guard let grid = videoGridTHW, videoIndex < grid.count else { break }
                    (t, h, w) = grid[videoIndex].values
                    videoIndex += 1
                    remainVideos -= 1
                    ed = edVideo
                }

                let llmGridT = t
                let llmGridH = h / spatialMergeSize
                let llmGridW = w / spatialMergeSize

                // Calculate starting index
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                // Add text tokens before this visual block
                let textLen = ed - st
                if textLen > 0 {
                    var index = MLXArray(0 ..< textLen).reshaped([1, textLen])
                    index = broadcast(index, to: [3, textLen])
                    index = index + MLXArray(stIdx)
                    llmPosIdsList.append(index)
                }

                // Add 3D position IDs for visual tokens
                // Python: mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                // Adds offset to ALL three dimensions!
                var tIndex = MLXArray(0 ..< llmGridT).reshaped([llmGridT, 1])
                tIndex = broadcast(tIndex, to: [llmGridT, llmGridH * llmGridW])
                tIndex = tIndex.flattened()

                var hIndex = MLXArray(0 ..< llmGridH).reshaped([1, llmGridH, 1])
                hIndex = broadcast(hIndex, to: [llmGridT, llmGridH, llmGridW])
                hIndex = hIndex.flattened()

                var wIndex = MLXArray(0 ..< llmGridW).reshaped([1, 1, llmGridW])
                wIndex = broadcast(wIndex, to: [llmGridT, llmGridH, llmGridW])
                wIndex = wIndex.flattened()

                let visualPosIds = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLen + stIdx)
                llmPosIdsList.append(visualPosIds)

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Add remaining text tokens after last visual block
            if st < inputTokens.count {
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                let textLen = inputTokens.count - st
                var tIndex = MLXArray(0 ..< textLen).reshaped([1, textLen])
                tIndex = broadcast(tIndex, to: [3, textLen])
                llmPosIdsList.append(tIndex + MLXArray(stIdx))
            }

            // Concatenate all position IDs for this batch item
            if !llmPosIdsList.isEmpty {
                let llmPositions = concatenated(llmPosIdsList, axis: 1)  // [3, seq]

                // Update position_ids for this batch
                let expandedMask = broadcast(
                    mask[batchIdx, 0...][.newAxis, .newAxis, 0...], to: [3, 1, seqLength])
                let expandedPositions = llmPositions[0..., .newAxis, 0...]
                let newPositions = `where`(
                    expandedMask, expandedPositions,
                    positionIds[0..., batchIdx ..< batchIdx + 1, 0...])

                // Replace this batch's position IDs (assumes batch size = 1)
                positionIds = newPositions

                let maxPosId = llmPositions.max().item(Int.self)
                mropePositionDeltas.append(maxPosId + 1 - inputTokens.count)
            }
        }

        // Python always returns deltas array (zeros for text-only, computed values for multimodal)
        let deltas: MLXArray
        if mropePositionDeltas.isEmpty {
            deltas = MLXArray.zeros([batchSize], dtype: .int32)
        } else {
            deltas = MLXArray(mropePositionDeltas.map { Int32($0) })
        }
        return (positionIds, deltas)
    }
}

// MARK: - Model

public final class Qwen3VLBackbone: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen3VLLanguage.LanguageModel

    public let config: Qwen3VLConfiguration

    public init(_ config: Qwen3VLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config)
    }

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = (imageMask .|| videoMask)

        let nImageTokens = specialMask.sum().item(Int.self)

        let mask2D = specialMask                                   // [B, S] bool (pre-expand)
        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen3VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }

        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)

        // Scatter image features into their token slots on-device: per-token feature
        // index = cumsum(mask)-1, gather one feature row per position, select it where
        // the mask is set (else keep the text embedding). Replaces a host round-trip
        // (nonZero → asArray(Bool) over seq×hidden → Swift index build → scatter).
        guard imageFeatures.dim(0) > 0 else { return (inputEmbeds, visualMask) }
        let b = inputEmbeds.dim(0), s = inputEmbeds.dim(1), h = inputEmbeds.dim(2)
        let flatMask = mask2D.asType(.int32).reshaped([-1])        // [B*S]
        let seg = maximum(cumsum(flatMask, axis: 0) - 1, MLXArray(Int32(0)))
        let gathered = imageFeatures[seg].reshaped([b, s, h])      // [B, S, H]
        let maskF = mask2D.asType(inputEmbeds.dtype).reshaped([b, s, 1])
        let result = inputEmbeds * (1 - maskF) + gathered * maskF
        return (result, visualMask)
    }

    private func combinedFrames(
        imageFrames: [THW]?,
        videoFrames: [THW]?
    ) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    /// Vision-tower + image-feature-merge results shared by ``prepare`` (logits) and
    /// ``lastHiddenState`` (embeddings).
    private struct LanguageInputs {
        let inputIds: MLXArray
        let inputEmbeddings: MLXArray?
        let visualMask: MLXArray?
        let deepstackEmbeds: [MLXArray]?
        let pixelValues: MLXArray?
        let imageFrames: [THW]?
        let videoFrames: [THW]?
    }

    private func buildLanguageInputs(_ input: LMInput) throws -> LanguageInputs {
        let inputIds = input.text.tokens

        var pixelValues: MLXArray?
        var imageFrames: [THW]? = nil
        var videoFrames: [THW]? = nil

        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }

        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoFrames = video.frames
        }

        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray? = nil
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]? = nil

        if let pixelValues,
            let framesList = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, deepstackOutputs) = visionModel(pixelValues, gridTHW: framesList)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = framesList.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let featureSlices = visionHidden.split(indices: splitIndices)
            let flattenedFeatures = concatenated(featureSlices).asType(textEmbeds.dtype)

            let (mergedEmbeds, mask) = try mergeInputIdsWithImageFeatures(
                imageFeatures: flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex)

            inputEmbeddings = mergedEmbeds
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let splitIndices = cumulativeSplitIndices(from: splits)
                    let slices = layerFeatures.split(indices: splitIndices)
                    let concatenatedSlices = concatenated(slices).asType(textEmbeds.dtype)
                    return concatenatedSlices
                }
            }
        }

        return LanguageInputs(
            inputIds: inputIds, inputEmbeddings: inputEmbeddings, visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds, pixelValues: pixelValues,
            imageFrames: imageFrames, videoFrames: videoFrames)
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state _: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        let li = try buildLanguageInputs(input)
        let typedCache = castCache(cache)

        let languageOutput = languageModel(
            li.inputIds,
            cache: typedCache,
            inputEmbeddings: li.inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualMask: li.visualMask,
            deepstackEmbeds: li.deepstackEmbeds,
            pixelValues: li.pixelValues,
            imageGridTHW: li.imageFrames,
            videoGridTHW: li.videoFrames)

        return .logits(languageOutput)
    }

    /// The model's `last_hidden_state` (pre-`lm_head`), `[batch, seq, hidden]`, for the
    /// embedding head. Same prefill path as ``prepare(_:cache:windowSize:)`` but returns the
    /// trunk output instead of vocabulary logits. (Local vendoring addition — see VENDORED.md.)
    public func lastHiddenState(_ input: LMInput, cache: [any KVCache]) throws -> MLXArray {
        let li = try buildLanguageInputs(input)
        let typedCache = castCache(cache)

        return languageModel.hiddenStates(
            li.inputIds,
            cache: typedCache,
            inputEmbeddings: li.inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualMask: li.visualMask,
            deepstackEmbeds: li.deepstackEmbeds,
            pixelValues: li.pixelValues,
            imageGridTHW: li.imageFrames,
            videoGridTHW: li.videoFrames)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)

        let result = languageModel(
            inputs,
            cache: typedCache,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        ).logits
        return result
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        for (key, value) in weights {
            var newKey = key

            if newKey.contains("model") {
                if newKey.contains("model.visual") {
                    newKey = newKey.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if newKey.contains("model.language_model") {
                    newKey = newKey.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                }
            } else if newKey.contains("lm_head") {
                newKey = newKey.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if config.textConfiguration.tieWordEmbeddings && newKey.contains(".lm_head.") {
                // if using tieWordEmbeddings omit these keys as they will not be consumed
                continue
            }

            adjusted[newKey] = value
        }

        let sanitized = visionModel.sanitize(weights: adjusted)
        return sanitized
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen3VLBackbone {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

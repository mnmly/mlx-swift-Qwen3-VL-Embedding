import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var model = DemoModel()
    @State private var showImageImporter = false
    @State private var showEmbedFolderPicker = false
    @State private var showRerankFolderPicker = false

    var body: some View {
        VStack(spacing: 0) {
            HSplitView {
                embeddingPane
                rerankerPane
            }
            Divider()
            statusBar
        }
    }

    // MARK: - Model picker (Finder dialog + bookmark, or download)

    private func modelSection(
        url: URL?, repoId: String, choose: @escaping () -> Void, download: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: url == nil ? "folder.badge.questionmark" : "folder.fill")
                    .foregroundStyle(url == nil ? Color.secondary : Color.green)
                Text(url?.lastPathComponent ?? "No model selected")
                    .lineLimit(1).truncationMode(.middle)
                    .foregroundStyle(url == nil ? .secondary : .primary)
            }
            HStack {
                Button("Choose folder…", action: choose)
                Button("Download \(repoId.split(separator: "/").last.map(String.init) ?? repoId)") {
                    download()
                }
                .disabled(model.downloadProgress != nil)
            }
            if let p = model.downloadProgress {
                ProgressView(value: p) {
                    Text(model.downloadStatus).font(.caption)
                }
            } else if !model.downloadStatus.isEmpty {
                Text(model.downloadStatus).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Embeddings

    private var embeddingPane: some View {
        Form {
            Section("Embedding model") {
                modelSection(
                    url: model.embeddingModelURL, repoId: model.embeddingRepoId,
                    choose: { showEmbedFolderPicker = true },
                    download: { model.downloadEmbeddingModel() })
            }
            Section("Texts (one per line)") {
                TextEditor(text: $model.textsText).font(.body.monospaced()).frame(minHeight: 90)
            }
            Section("Images (embedded in the same space as the texts)") {
                Button("Add image…") { showImageImporter = true }
                if !model.images.isEmpty {
                    ScrollView(.horizontal) {
                        HStack(spacing: 10) {
                            ForEach(Array(model.images.enumerated()), id: \.offset) { idx, img in
                                thumbnail(img.image, index: idx)
                            }
                        }
                    }
                }
            }
            Section {
                Button("Embed → cross-modal cosine similarity") { model.embed() }
                    .disabled(model.isBusy || model.embeddingModelURL == nil)
            }
            if !model.similarity.isEmpty {
                if let pair = bestImageSentencePair() {
                    Section("Best image ↔ sentence") {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(label(pair.image)).foregroundStyle(.tint).lineLimit(1)
                            Label(label(pair.text), systemImage: "text.quote").lineLimit(2)
                            Text(String(format: "cosine %.3f", pair.score))
                                .font(.caption.monospaced()).foregroundStyle(.secondary)
                        }
                    }
                }
                Section("Cosine similarity") {
                    similarityGrid(model.similarity)
                    legend(model.itemLabels)
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 400)
        .fileImporter(
            isPresented: $showEmbedFolderPicker, allowedContentTypes: [.folder]
        ) { result in
            if case .success(let url) = result { model.setEmbeddingModel(url) }
        }
        .fileImporter(
            isPresented: $showImageImporter, allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { model.addImages(urls) }
        }
    }

    private func thumbnail(_ cg: CGImage, index: Int) -> some View {
        VStack(spacing: 2) {
            Image(decorative: cg, scale: 1)
                .resizable().scaledToFill()
                .frame(width: 56, height: 56).clipped()
                .clipShape(RoundedRectangle(cornerRadius: 4))
            Button(role: .destructive) { model.removeImage(at: index) } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.borderless)
        }
    }

    private func similarityGrid(_ matrix: [[Float]]) -> some View {
        Grid(horizontalSpacing: 6, verticalSpacing: 4) {
            ForEach(matrix.indices, id: \.self) { i in
                GridRow {
                    Text("\(i)").foregroundStyle(.secondary).frame(width: 16)
                    ForEach(matrix[i].indices, id: \.self) { j in
                        Text(String(format: "%.2f", matrix[i][j]))
                            .font(.caption.monospaced())
                            .frame(width: 40).padding(2)
                            .background(heat(matrix[i][j]))
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                }
            }
        }
    }

    private func legend(_ labels: [String]) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            ForEach(labels.indices, id: \.self) { i in
                Text("\(i)  \(labels[i])").font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }

    /// The single highest-scoring (image, sentence) cross-modal pair, if both are present.
    /// Images are the items whose label is prefixed with the picture glyph (see `embed`).
    private func bestImageSentencePair() -> (image: Int, text: Int, score: Float)? {
        let labels = model.itemLabels
        let imageIdx = labels.indices.filter { labels[$0].hasPrefix("🖼") }
        let textIdx = labels.indices.filter { !labels[$0].hasPrefix("🖼") }
        guard !imageIdx.isEmpty, !textIdx.isEmpty else { return nil }
        var best: (image: Int, text: Int, score: Float)?
        for im in imageIdx where model.similarity.indices.contains(im) {
            for t in textIdx where model.similarity[im].indices.contains(t) {
                let s = model.similarity[im][t]
                if best == nil || s > best!.score { best = (image: im, text: t, score: s) }
            }
        }
        return best
    }

    private func label(_ i: Int) -> String {
        model.itemLabels.indices.contains(i) ? model.itemLabels[i] : "\(i)"
    }

    private func heat(_ v: Float) -> Color {
        Color(hue: 0.58, saturation: Double(max(0, min(1, v))) * 0.7, brightness: 0.95)
    }

    // MARK: - Reranker

    private var rerankerPane: some View {
        Form {
            Section("Reranker model") {
                modelSection(
                    url: model.rerankerModelURL, repoId: model.rerankerRepoId,
                    choose: { showRerankFolderPicker = true },
                    download: { model.downloadRerankerModel() })
            }
            Section("Query") {
                TextField("query", text: $model.query).textFieldStyle(.roundedBorder)
            }
            Section("Documents (one per line)") {
                TextEditor(text: $model.documentsText).font(.body.monospaced()).frame(minHeight: 120)
                Button("Rank by relevance") { model.rerank() }
                    .disabled(model.isBusy || model.rerankerModelURL == nil)
            }
            if !model.ranked.isEmpty {
                Section("Ranked (best first)") {
                    ForEach(model.ranked.indices, id: \.self) { k in
                        let entry = model.ranked[k]
                        HStack {
                            Text(String(format: "%.4f", entry.score))
                                .font(.body.monospaced()).foregroundStyle(.tint)
                            Text(model.documents[safe: entry.index] ?? "").lineLimit(1)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 400)
        .fileImporter(
            isPresented: $showRerankFolderPicker, allowedContentTypes: [.folder]
        ) { result in
            if case .success(let url) = result { model.setRerankerModel(url) }
        }
    }

    // MARK: - Status

    private var statusBar: some View {
        HStack {
            if model.isBusy { ProgressView().controlSize(.small) }
            Text(model.status).font(.callout)
            Spacer()
            Text(model.memory).font(.caption.monospaced()).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

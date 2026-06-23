import XCTest

@testable import MLXQwen3VLEmbedding

/// Phase 0 smoke: the package builds and links. Model-backed parity tests
/// (which need multi-GB weights) land in later phases and skip when weights
/// are absent.
final class ScaffoldTests: XCTestCase {
    func testVersionPresent() {
        XCTAssertFalse(MLXQwen3VLEmbedding.version.isEmpty)
    }

    func testTaskCases() {
        XCTAssertEqual(Qwen3VLTask.allCases.count, 2)
    }
}

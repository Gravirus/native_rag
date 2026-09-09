from pathlib import Path

import numpy as np
import pytest

from native_rag.backend import BackendCapabilities
from native_rag.documents import DocumentChunk
from native_rag.index import DocumentIndex


class DummyGenerator:
    model_path = Path("dummy-model")
    capabilities = BackendCapabilities(supports_native_dense_features=True)
    dense_feature_descriptor = {
        "provider": "model_native",
        "model_fingerprint": "dummy",
        "architecture": "dummy",
        "feature_layer": 0,
        "pooling": "mean",
        "normalization": "none",
        "dimension": 2,
    }

    def native_dense_features(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


def test_index_roundtrip(tmp_path: Path) -> None:
    chunks = [DocumentChunk(0, "doc.md", "hello documentation", 0, 19, "Intro")]
    index = DocumentIndex(chunks, np.array([[1.0, 0.0]], dtype=np.float32), {"model_path": "test"})
    directory = tmp_path / "index"
    index.save(directory)
    loaded = DocumentIndex.load(directory)
    assert loaded.chunks == chunks
    assert loaded.dense is not None
    assert loaded.search("documentation", top_k=1)[0].chunk.id == 0


def test_index_builds_embeddings_through_backend(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# Guide\n\nUse the documentation.", encoding="utf-8")
    index = DocumentIndex.build(tmp_path, generator=DummyGenerator(), chunk_chars=100, overlap_chars=10)
    assert index.dense is not None
    assert index.metadata["embedding_dim"] == 2
    assert index.metadata["dense"]["provider"] == "model_native"


def test_index_without_native_features_is_model_independent(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# Guide\n\nUse the documentation.", encoding="utf-8")
    index = DocumentIndex.build(tmp_path)

    assert index.dense is None
    assert index.metadata["dense"] is None


def test_index_rejects_incompatible_native_dense_generator(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# Guide\n\nUse the documentation.", encoding="utf-8")
    index = DocumentIndex.build(tmp_path, generator=DummyGenerator())
    mismatched = DummyGenerator()
    mismatched.dense_feature_descriptor = {
        **DummyGenerator.dense_feature_descriptor,
        "model_fingerprint": "different-model",
    }

    with pytest.raises(ValueError, match="does not match the index"):
        index.search("documentation", top_k=1, generator=mismatched)


def test_search_expands_neighbors_without_crossing_documents() -> None:
    chunks = [
        DocumentChunk(0, "a.md", "before", 0, 6, "A"),
        DocumentChunk(1, "a.md", "answer", 7, 13, "A"),
        DocumentChunk(2, "a.md", "after", 14, 19, "A"),
        DocumentChunk(3, "b.md", "other", 0, 5, "B"),
    ]
    index = DocumentIndex(chunks)
    hits = index.search("answer", top_k=1, neighbor_radius=1)
    assert [hit.chunk.id for hit in hits] == [0, 1, 2]
    assert hits[1].channel == "lexical"
    assert hits[0].channel.endswith("neighbor")


def test_search_separates_candidate_pool_from_context_budget() -> None:
    chunks = [
        DocumentChunk(0, "a.md", "alpha", 0, 5),
        DocumentChunk(1, "a.md", "answer alpha", 6, 18),
        DocumentChunk(2, "a.md", "beta", 19, 23),
        DocumentChunk(3, "a.md", "gamma", 24, 29),
    ]
    index = DocumentIndex(chunks)

    hits = index.search(
        "answer",
        top_k=1,
        candidate_k=64,
        neighbor_radius=1,
        max_context_chunks=3,
    )

    assert len(hits) <= 3
    assert 1 in [hit.chunk.id for hit in hits]


def test_search_can_coarsen_adjacent_chunks_for_context() -> None:
    chunks = [
        DocumentChunk(0, "a.md", "setup", 0, 5, "A"),
        DocumentChunk(1, "a.md", "answer detail", 6, 19, "A"),
        DocumentChunk(2, "a.md", "follow up", 20, 29, "A"),
        DocumentChunk(3, "a.md", "tail", 30, 34, "A"),
    ]
    index = DocumentIndex(chunks)

    hits = index.search(
        "answer",
        top_k=1,
        candidate_k=4,
        span_size=2,
        neighbor_radius=1,
        max_context_chunks=1,
    )

    assert len(hits) == 1
    assert hits[0].chunk.text == "setup\n\nanswer detail"

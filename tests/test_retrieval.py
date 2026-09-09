import numpy as np

from native_rag.documents import DocumentChunk
from native_rag.retrieval import (
    DenseIndex,
    HybridRetriever,
    LexicalIndex,
    ScoredChunk,
    coarsen_chunks,
    coarsen_ranked,
    expand_neighbors,
    rrf_fuse,
    select_context_spans,
    select_adaptive_spans,
)


def chunks():
    return [
        DocumentChunk(0, "a.md", "database backup retention policy", 0, 33),
        DocumentChunk(1, "b.md", "network timeout and retry policy", 0, 33),
        DocumentChunk(2, "c.md", "unrelated cooking recipe", 0, 23),
    ]


def test_bm25_returns_relevant_chunk() -> None:
    result = LexicalIndex(chunks()).search("backup retention", top_k=1)
    assert result[0].chunk.id == 0


def test_bm25_normalizes_hyphenated_terms() -> None:
    items = [DocumentChunk(0, "doc.md", "request timeout policy", 0, 22)]
    result = LexicalIndex(items).search("request-timeout", top_k=1)
    assert [item.chunk.id for item in result] == [0]


def test_allowed_ids_are_hard_filter() -> None:
    result = LexicalIndex(chunks()).search("backup retention", top_k=3, allowed_ids={2})
    assert [item.chunk.id for item in result] == []


def test_rrf_is_deterministic() -> None:
    items = chunks()
    first = [ScoredChunk(items[1], 1.0, "lexical"), ScoredChunk(items[0], 0.9, "lexical")]
    second = [ScoredChunk(items[0], 1.0, "dense"), ScoredChunk(items[1], 0.9, "dense")]
    result = rrf_fuse([first, second], top_k=2, rrf_k=60)
    assert {item.chunk.id for item in result} == {0, 1}
    assert result[0].channel == "dense+lexical"


def test_hybrid_retriever_respects_allowlist() -> None:
    items = chunks()
    dense = DenseIndex(items, np.eye(3, dtype=np.float32))
    retriever = HybridRetriever(LexicalIndex(items), dense)
    result = retriever.search("backup", top_k=3, query_embedding=np.array([1, 0, 0]), allowed_ids={2})
    assert [item.chunk.id for item in result] == [2]


def test_neighbor_budget_keeps_ranked_anchors() -> None:
    items = [
        DocumentChunk(0, "doc.md", "zero", 0, 4),
        DocumentChunk(1, "doc.md", "one", 5, 8),
        DocumentChunk(2, "doc.md", "two", 9, 12),
        DocumentChunk(3, "doc.md", "three", 13, 18),
        DocumentChunk(4, "doc.md", "four", 19, 23),
    ]
    ranked = [ScoredChunk(items[1], 10.0, "lexical"), ScoredChunk(items[3], 9.0, "lexical")]

    result = expand_neighbors(ranked, items, radius=2, max_chunks=2)

    assert [item.chunk.id for item in result] == [1, 3]


def test_zero_radius_uses_the_entire_context_budget_for_anchors() -> None:
    items = [DocumentChunk(index, "doc.md", str(index), index, index + 1) for index in range(4)]
    ranked = [ScoredChunk(item, 10.0 - item.id, "lexical") for item in items]

    result = select_context_spans(ranked, items, radius=0, max_chunks=3)

    assert [item.chunk.id for item in result] == [0, 1, 2]


def test_coarsening_does_not_cross_document_boundaries() -> None:
    items = [
        DocumentChunk(0, "a.md", "first", 0, 5, "A"),
        DocumentChunk(1, "a.md", "second", 6, 12, "A"),
        DocumentChunk(2, "a.md", "third", 13, 18, "A"),
        DocumentChunk(3, "b.md", "other", 0, 5, "B"),
    ]

    coarse, mapping = coarsen_chunks(items, group_size=2)

    assert [(chunk.source, chunk.text) for chunk in coarse] == [
        ("a.md", "first\n\nsecond"),
        ("a.md", "third"),
        ("b.md", "other"),
    ]
    assert mapping == {0: 0, 1: 0, 2: 1, 3: 2}
    assert coarse[0].heading == "A"


def test_coarsened_ranking_merges_hits_deterministically() -> None:
    items = [
        DocumentChunk(0, "doc.md", "alpha", 0, 5),
        DocumentChunk(1, "doc.md", "answer alpha", 6, 18),
        DocumentChunk(2, "doc.md", "other", 19, 24),
    ]
    coarse, mapping = coarsen_chunks(items, group_size=2)
    ranked = [
        ScoredChunk(items[1], 2.0, "lexical"),
        ScoredChunk(items[0], 1.0, "lexical"),
        ScoredChunk(items[2], 0.5, "lexical"),
    ]

    result = coarsen_ranked(ranked, mapping, coarse)

    assert [item.chunk.id for item in result] == [0, 1]
    assert result[0].score == 2.05
    assert result[0].channel == "lexical+span"


def test_adaptive_spans_preserve_budget_and_merge_only_selected_neighbors() -> None:
    items = [
        DocumentChunk(0, "doc.md", "zero", 0, 4),
        DocumentChunk(1, "doc.md", "one", 5, 8),
        DocumentChunk(2, "doc.md", "two", 9, 12),
        DocumentChunk(3, "doc.md", "three", 13, 18),
    ]
    ranked = [ScoredChunk(items[1], 10.0, "lexical")]

    result = select_adaptive_spans(ranked, items, span_size=2, radius=1, max_small_chunks=3)

    assert [item.chunk.text for item in result] == ["zero\n\none", "two"]
    assert sum(item.chunk.text.count("\n\n") + 1 for item in result) == 3
    assert all("three" not in item.chunk.text for item in result)


def test_adaptive_spans_keep_the_first_chunk_after_a_gap() -> None:
    items = [
        DocumentChunk(0, "doc.md", "zero", 0, 4),
        DocumentChunk(1, "doc.md", "one", 5, 8),
        DocumentChunk(2, "doc.md", "two", 9, 12),
    ]
    ranked = [
        ScoredChunk(items[0], 10.0, "lexical"),
        ScoredChunk(items[2], 9.0, "lexical"),
    ]

    result = select_adaptive_spans(ranked, items, span_size=2, radius=0, max_small_chunks=2)

    assert [item.chunk.text for item in result] == ["zero", "two"]

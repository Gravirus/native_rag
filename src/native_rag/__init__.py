"""Document-focused Native RAG."""

from .documents import DocumentChunk, load_chunks
from .index import DocumentIndex
from .retrieval import (
    HybridRetriever,
    LexicalIndex,
    coalesce_selected_chunks,
    coarsen_chunks,
    coarsen_ranked,
    expand_neighbors,
    rrf_fuse,
    select_adaptive_spans,
    select_context_spans,
)

__all__ = [
    "DocumentChunk",
    "DocumentIndex",
    "HybridRetriever",
    "LexicalIndex",
    "coalesce_selected_chunks",
    "coarsen_chunks",
    "coarsen_ranked",
    "expand_neighbors",
    "load_chunks",
    "rrf_fuse",
    "select_adaptive_spans",
    "select_context_spans",
]

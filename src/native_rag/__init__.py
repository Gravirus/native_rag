"""Document-focused Native RAG."""

from .documents import DocumentChunk, load_chunks
from .index import DocumentIndex
from .prompt import PromptDocument, PromptSpec, build_prompt_spec
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
    "BackendCapabilities",
    "BackendCapabilityError",
    "CompiledPrompt",
    "DocumentChunk",
    "DocumentIndex",
    "GeneratorBackend",
    "GenericHFBackend",
    "GenerationResult",
    "HybridRetriever",
    "LexicalIndex",
    "PromptDocument",
    "PromptSpec",
    "TransformersBackend",
    "build_prompt_spec",
    "coalesce_selected_chunks",
    "coarsen_chunks",
    "coarsen_ranked",
    "expand_neighbors",
    "load_chunks",
    "rrf_fuse",
    "select_adaptive_spans",
    "select_context_spans",
    "load_backend",
]


def __getattr__(name: str):
    """Load model-facing APIs lazily for lightweight retrieval-only use."""
    backend_names = {
        "BackendCapabilities",
        "BackendCapabilityError",
        "CompiledPrompt",
        "GeneratorBackend",
        "GenerationResult",
    }
    model_names = {"GenericHFBackend", "TransformersBackend", "load_backend"}
    if name in backend_names:
        from . import backend

        return getattr(backend, name)
    if name in model_names:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)

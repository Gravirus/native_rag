from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .documents import DocumentChunk, load_chunks
from .retrieval import (
    DenseIndex,
    HybridRetriever,
    LexicalIndex,
    ScoredChunk,
    select_adaptive_spans,
    select_context_spans,
)

if TYPE_CHECKING:
    from .backend import GeneratorBackend


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1


class DocumentIndex:
    def __init__(self, chunks: list[DocumentChunk], embeddings: np.ndarray | None = None,
                 metadata: dict | None = None):
        self.chunks = list(chunks)
        self.lexical = LexicalIndex(self.chunks)
        self.dense = DenseIndex(self.chunks, embeddings) if embeddings is not None else None
        self.metadata = metadata or {}

    @classmethod
    def build(
        cls,
        docs_dir: Path,
        generator: GeneratorBackend | None = None,
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
        native_dense: bool | None = None,
    ) -> "DocumentIndex":
        """Build an index, optionally using native features from one generator."""
        if native_dense is True and generator is None:
            raise ValueError("native dense indexing requires a generator backend")
        chunks = load_chunks(docs_dir, chunk_chars, overlap_chars)
        feature_provider = None
        dense_descriptor = None
        if generator is not None and native_dense is not False:
            capabilities = getattr(generator, "capabilities", None)
            supports_native = bool(
                getattr(capabilities, "supports_native_dense_features", False)
            )
            if supports_native:
                feature_provider = getattr(generator, "native_dense_features", None)
                dense_descriptor = getattr(generator, "dense_feature_descriptor", None)
                if not callable(feature_provider) or dense_descriptor is None:
                    raise ValueError("native dense capability is missing its feature contract")
            elif native_dense is True:
                raise ValueError("the selected generator does not support native dense features")
        embeddings = feature_provider([chunk.text for chunk in chunks]) if feature_provider else None
        if embeddings is not None and dense_descriptor is not None:
            dense_descriptor = dict(dense_descriptor)
            dense_descriptor["dimension"] = int(embeddings.shape[1])
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "document_root": str(Path(docs_dir).resolve()),
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
            "embedding_dim": int(embeddings.shape[1]) if embeddings is not None else None,
            "model_path": str(generator.model_path) if generator and hasattr(generator, "model_path") else None,
            "dense": dense_descriptor if embeddings is not None else None,
        }
        return cls(chunks, embeddings, metadata)

    def retriever(self, rrf_k: int = 60) -> HybridRetriever:
        return HybridRetriever(self.lexical, self.dense, rrf_k=rrf_k)

    def search(
        self,
        query: str,
        top_k: int = 8,
        generator: GeneratorBackend | None = None,
        allowed_ids: set[int] | None = None,
        rrf_k: int = 60,
        neighbor_radius: int = 0,
        max_context_chunks: int | None = None,
        candidate_k: int | None = None,
        span_size: int = 1,
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        if candidate_k is not None and candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if span_size <= 0:
            raise ValueError("span_size must be positive")
        if neighbor_radius < 0:
            raise ValueError("neighbor_radius must be non-negative")
        context_budget = max_context_chunks
        if neighbor_radius and context_budget is None:
            context_budget = top_k * (2 * neighbor_radius + 1)
        retrieval_k = top_k
        if neighbor_radius:
            retrieval_k = candidate_k or max(64, context_budget * 2, top_k)
        elif candidate_k is not None:
            retrieval_k = candidate_k
        query_embedding = None
        if self.dense is not None and generator is not None:
            capabilities = getattr(generator, "capabilities", None)
            supports_native = bool(
                getattr(capabilities, "supports_native_dense_features", False)
            )
            if supports_native:
                actual = getattr(generator, "dense_feature_descriptor", None)
                expected = self.metadata.get("dense")
                if not expected or expected.get("provider") != "model_native":
                    raise ValueError(
                        "dense index has no verifiable native feature contract; rebuild the index"
                    )
                mismatches = {
                    key: (expected.get(key), actual.get(key) if actual else None)
                    for key in (
                        "model_fingerprint",
                        "architecture",
                        "feature_layer",
                        "pooling",
                        "normalization",
                        "dimension",
                    )
                    if expected.get(key) != (actual or {}).get(key)
                }
                if mismatches:
                    raise ValueError(
                        "native dense feature contract does not match the index: "
                        f"{mismatches}"
                    )
                features = getattr(generator, "native_dense_features", None)
                if not callable(features):
                    raise ValueError("generator advertises native dense features but cannot produce them")
                query_embedding = features([query])[0]
        hits = self.retriever(rrf_k).search(query, retrieval_k, query_embedding, allowed_ids)
        if span_size > 1:
            span_budget = context_budget if context_budget is not None else top_k
            hits = select_adaptive_spans(
                hits,
                self.chunks,
                span_size=span_size,
                radius=neighbor_radius,
                max_small_chunks=span_budget * span_size,
            )
        elif neighbor_radius:
            hits = select_context_spans(hits, self.chunks, neighbor_radius, context_budget)
        elif max_context_chunks is not None:
            hits = hits[:max_context_chunks]
        elif candidate_k is not None:
            hits = hits[:top_k]
        return hits

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        chunks_path = directory / "chunks.json"
        chunks_path.write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        if self.dense is not None:
            np.save(directory / "embeddings.npy", self.dense.embeddings)
        manifest = dict(self.metadata)
        manifest.update({
            "schema_version": SCHEMA_VERSION,
            "chunk_count": len(self.chunks),
            "has_embeddings": self.dense is not None,
            "content_sha256": self._content_hash(),
        })
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DocumentIndex":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        schema_version = manifest.get("schema_version")
        if schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
            raise ValueError("unsupported index schema version")
        raw_chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**item) for item in raw_chunks]
        embeddings = np.load(directory / "embeddings.npy") if manifest.get("has_embeddings") else None
        metadata = dict(manifest)
        if schema_version == LEGACY_SCHEMA_VERSION:
            # Old dense vectors cannot be proven compatible with a future
            # generator. Keep the chunks/BM25 usable, but force a rebuild for
            # model-native dense retrieval.
            metadata["schema_version"] = SCHEMA_VERSION
            if embeddings is not None:
                metadata.setdefault(
                    "dense",
                    {
                        "provider": "legacy",
                        "model_path": metadata.get("model_path"),
                        "dimension": int(embeddings.shape[1]),
                    },
                )
        index = cls(chunks, embeddings, metadata)
        if manifest.get("content_sha256") != index._content_hash():
            raise ValueError("index content checksum mismatch")
        return index

    def _content_hash(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(chunk.source.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

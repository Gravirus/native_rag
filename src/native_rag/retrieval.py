from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .documents import DocumentChunk


TOKEN_RE = re.compile(r"(?u)[^\W_]{2,}(?:[-'][^\W_]{2,})*")
DEFAULT_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "into",
    "как", "что", "это", "для", "при", "или", "из", "на", "по", "не", "до", "так",
}


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    terms: list[str] = []
    for token in TOKEN_RE.findall(normalized):
        terms.append(token)
        if "-" in token or "'" in token:
            terms.extend(part for part in re.split(r"[-']", token) if len(part) >= 2)
    return terms


@dataclass(frozen=True)
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    channel: str


def coarsen_chunks(
    chunks: Sequence[DocumentChunk],
    group_size: int = 2,
) -> tuple[list[DocumentChunk], dict[int, int]]:
    """Group adjacent retrieval chunks into larger context spans.

    Groups are formed independently per source, so a context span never
    crosses a document boundary. The returned mapping translates every small
    chunk id to its coarse-span id.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source].append(chunk)
    coarse: list[DocumentChunk] = []
    small_to_coarse: dict[int, int] = {}
    for source in sorted(by_source):
        source_chunks = sorted(by_source[source], key=lambda chunk: (chunk.start_char, chunk.id))
        for start in range(0, len(source_chunks), group_size):
            members = source_chunks[start:start + group_size]
            coarse_id = len(coarse)
            coarse.append(DocumentChunk(
                coarse_id,
                source,
                "\n\n".join(member.text for member in members),
                members[0].start_char,
                members[-1].end_char,
                next((member.heading for member in members if member.heading), None),
            ))
            for member in members:
                small_to_coarse[member.id] = coarse_id
    return coarse, small_to_coarse


def coarsen_ranked(
    ranked: Sequence[ScoredChunk],
    small_to_coarse: dict[int, int],
    coarse_chunks: Sequence[DocumentChunk],
) -> list[ScoredChunk]:
    """Aggregate ranked small chunks into deterministically scored spans."""
    grouped: dict[int, list[ScoredChunk]] = defaultdict(list)
    for item in ranked:
        coarse_id = small_to_coarse.get(item.chunk.id)
        if coarse_id is not None:
            grouped[coarse_id].append(item)
    coarse_by_id = {chunk.id: chunk for chunk in coarse_chunks}
    scored: list[ScoredChunk] = []
    for coarse_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: (-item.score, item.chunk.id))
        score = ordered[0].score + (0.05 * ordered[1].score if len(ordered) > 1 else 0.0)
        scored.append(ScoredChunk(coarse_by_id[coarse_id], score, f"{ordered[0].channel}+span"))
    return sorted(scored, key=lambda item: (-item.score, item.chunk.source, item.chunk.start_char, item.chunk.id))


def coalesce_selected_chunks(
    selected: Sequence[ScoredChunk],
    chunks: Sequence[DocumentChunk],
    group_size: int = 2,
) -> list[ScoredChunk]:
    """Merge only selected adjacent chunks into bounded variable-size spans."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source].append(chunk)
    for source_chunks in by_source.values():
        source_chunks.sort(key=lambda chunk: (chunk.start_char, chunk.id))
    positions = {
        chunk.id: (source, position)
        for source, source_chunks in by_source.items()
        for position, chunk in enumerate(source_chunks)
    }
    selected_by_id: dict[int, ScoredChunk] = {}
    for item in selected:
        if item.chunk.id in positions:
            selected_by_id.setdefault(item.chunk.id, item)

    output: list[ScoredChunk] = []
    next_id = 0
    for source in sorted(by_source):
        source_chunks = by_source[source]
        selected_positions = [
            position for position, chunk in enumerate(source_chunks)
            if chunk.id in selected_by_id
        ]
        run: list[int] = []
        for position in selected_positions + [None]:
            if position is not None and (not run or position == run[-1] + 1):
                run.append(position)
                continue
            if run:
                for start in range(0, len(run), group_size):
                    member_positions = run[start:start + group_size]
                    members = [source_chunks[position] for position in member_positions]
                    member_hits = [selected_by_id[member.id] for member in members]
                    ranked_members = sorted(
                        member_hits,
                        key=lambda item: (-item.score, item.chunk.id),
                    )
                    score = ranked_members[0].score
                    if len(ranked_members) > 1:
                        score += 0.05 * ranked_members[1].score
                    span = DocumentChunk(
                        next_id,
                        source,
                        "\n\n".join(member.text for member in members),
                        members[0].start_char,
                        members[-1].end_char,
                        next((member.heading for member in members if member.heading), None),
                    )
                    output.append(ScoredChunk(
                        span,
                        score,
                        f"{ranked_members[0].channel}+adaptive-span",
                    ))
                    next_id += 1
                run = []
                if position is not None:
                    run.append(position)
    return sorted(output, key=lambda item: (item.chunk.source, item.chunk.start_char, item.chunk.id))


def select_adaptive_spans(
    ranked: Sequence[ScoredChunk],
    chunks: Sequence[DocumentChunk],
    span_size: int = 2,
    radius: int = 0,
    max_small_chunks: int | None = None,
) -> list[ScoredChunk]:
    """Select small-chunk context, then coalesce only adjacent selected chunks.

    ``max_small_chunks`` is the hard context capacity.  It is deliberately
    measured in the indexed chunk units rather than returned spans: a distant
    anchor may remain a one-chunk span, while adjacent anchors can share a
    larger span.  This keeps token budget and retrieval recall independent
    from the number of output fragments.
    """
    if span_size <= 0:
        raise ValueError("span_size must be positive")
    selected = select_context_spans(ranked, chunks, radius, max_small_chunks)
    if span_size == 1:
        return selected
    return coalesce_selected_chunks(selected, chunks, span_size)


def select_context_spans(
    ranked: Sequence[ScoredChunk],
    chunks: Sequence[DocumentChunk],
    radius: int = 1,
    max_chunks: int | None = None,
) -> list[ScoredChunk]:
    """Select contiguous local spans around ranked candidate anchors.

    Retrieval decides *where* the answer is; this stage restores enough local
    document structure for the model to resolve references across chunk
    boundaries.  Candidate anchors can be deeper than the final context
    budget.  Spans never cross a source file, and the final result is emitted
    in document order.  If a span does not fit the remaining budget, its
    anchor is kept and the rest of the span is skipped.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if max_chunks is not None and max_chunks <= 0:
        return []
    by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source].append(chunk)
    for source_chunks in by_source.values():
        source_chunks.sort(key=lambda chunk: (chunk.start_char, chunk.id))

    positions: dict[int, tuple[str, int]] = {
        chunk.id: (source, position)
        for source, source_chunks in by_source.items()
        for position, chunk in enumerate(source_chunks)
    }

    candidates = []
    seen_ids: set[int] = set()
    for item in ranked:
        if item.chunk.id in positions and item.chunk.id not in seen_ids:
            candidates.append(item)
            seen_ids.add(item.chunk.id)

    if max_chunks is None:
        anchor_count = len(candidates)
    elif len(candidates) <= max_chunks:
        # There is no recall trade-off when the candidate pool already fits.
        anchor_count = len(candidates)
    else:
        # Reserve a very small anchor layer. In a normal 32-chunk window this
        # leaves one slot for radius 1 and two for radius 2. The active window
        # must keep broad recall; local expansion is a boundary repair, not a
        # reason to discard otherwise useful independent candidates.
        expansion_slots = 0 if radius == 0 else (
            max(1, max_chunks // 32) if radius == 1 else max(1, max_chunks // 16)
        )
        anchor_count = max(1, max_chunks - expansion_slots)
    anchors = candidates[:anchor_count]
    selected: dict[int, ScoredChunk] = {item.chunk.id: item for item in anchors}

    if radius:
        # Build a local candidate pool around the retained anchors. A chunk
        # can belong to multiple spans; prefer the closest/highest-ranked
        # anchor as its provenance.
        neighbours: dict[int, tuple[tuple[int, int, str, int, int], ScoredChunk]] = {}
        for anchor_rank, item in enumerate(anchors):
            source, position = positions[item.chunk.id]
            source_chunks = by_source[source]
            for neighbour_position in range(
                max(0, position - radius),
                min(len(source_chunks), position + radius + 1),
            ):
                neighbour = source_chunks[neighbour_position]
                if neighbour.id in selected:
                    continue
                distance = abs(neighbour_position - position)
                priority = (distance, anchor_rank, source, neighbour.start_char, neighbour.id)
                candidate = ScoredChunk(neighbour, item.score, f"{item.channel}+neighbor")
                previous = neighbours.get(neighbour.id)
                if previous is None or priority < previous[0]:
                    neighbours[neighbour.id] = (priority, candidate)

        ordered_neighbours = [
            item for _, item in sorted(neighbours.values(), key=lambda value: value[0])
        ]
        if max_chunks is None:
            selected.update({item.chunk.id: item for item in ordered_neighbours})
        else:
            remaining = max_chunks - len(selected)
            selected.update({item.chunk.id: item for item in ordered_neighbours[:max(0, remaining)]})

    ordered = sorted(
        selected.values(),
        key=lambda item: (item.chunk.source, item.chunk.start_char, item.chunk.id),
    )
    return ordered if max_chunks is None else ordered[:max_chunks]


def expand_neighbors(
    ranked: Sequence[ScoredChunk],
    chunks: Sequence[DocumentChunk],
    radius: int = 1,
    max_chunks: int | None = None,
) -> list[ScoredChunk]:
    """Backward-compatible name for :func:`select_context_spans`."""
    return select_context_spans(ranked, chunks, radius, max_chunks)


class LexicalIndex:
    """Small, deterministic BM25 index for a document-sized corpus."""

    def __init__(self, chunks: Sequence[DocumentChunk], stopwords: set[str] | None = None):
        self.chunks = list(chunks)
        self.stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_lengths: dict[int, int] = {}
        self.normalized_text: dict[int, str] = {}
        self.heading_terms: dict[int, set[str]] = {}
        for chunk in self.chunks:
            terms = [t for t in tokenize(chunk.text) if t not in self.stopwords]
            counts = Counter(terms)
            self.doc_lengths[chunk.id] = len(terms)
            self.normalized_text[chunk.id] = " ".join(terms)
            self.heading_terms[chunk.id] = {
                term for term in tokenize(chunk.heading or "") if term not in self.stopwords
            }
            for term, count in counts.items():
                self.postings[term][chunk.id] = count
        self.avgdl = sum(self.doc_lengths.values()) / max(1, len(self.chunks))

    def search(self, query: str, top_k: int = 8, allowed_ids: set[int] | None = None) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        query_terms = [t for t in tokenize(query) if t not in self.stopwords]
        query_counts = Counter(query_terms)
        scores: dict[int, float] = defaultdict(float)
        n_docs = max(1, len(self.chunks))
        k1, b = 1.5, 0.75
        for term, qtf in query_counts.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for chunk_id, tf in posting.items():
                if allowed_ids is not None and chunk_id not in allowed_ids:
                    continue
                length = self.doc_lengths[chunk_id]
                norm = 1.0 - b + b * length / max(1.0, self.avgdl)
                scores[chunk_id] += qtf * idf * tf * (k1 + 1.0) / (tf + k1 * norm)
        by_id = {chunk.id: chunk for chunk in self.chunks}
        normalized_query = " ".join(query_terms)
        for chunk_id in scores:
            if normalized_query and normalized_query in self.normalized_text[chunk_id]:
                scores[chunk_id] += 0.35
            heading_overlap = len(set(query_terms).intersection(self.heading_terms[chunk_id]))
            if heading_overlap:
                scores[chunk_id] += 0.45 * heading_overlap
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [ScoredChunk(by_id[i], float(score), "lexical") for i, score in ranked]


class DenseIndex:
    def __init__(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray):
        self.chunks = list(chunks)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.chunks):
            raise ValueError("embeddings must have shape [number_of_chunks, dimension]")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normalized = self.embeddings / np.clip(norms, 1e-8, None)

    def search(self, query_embedding: np.ndarray, top_k: int = 8, allowed_ids: set[int] | None = None) -> list[ScoredChunk]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query embedding dimension does not match index")
        query = query / max(float(np.linalg.norm(query)), 1e-8)
        scores = self.normalized @ query
        candidates = range(len(self.chunks)) if allowed_ids is None else [i for i, c in enumerate(self.chunks) if c.id in allowed_ids]
        ranked = sorted(candidates, key=lambda i: (-float(scores[i]), self.chunks[i].id))[:top_k]
        return [ScoredChunk(self.chunks[i], float(scores[i]), "dense") for i in ranked]


def rrf_fuse(rankings: Iterable[Sequence[ScoredChunk]], top_k: int = 8, rrf_k: int = 60) -> list[ScoredChunk]:
    if top_k <= 0:
        return []
    scores: dict[int, float] = defaultdict(float)
    chunks: dict[int, DocumentChunk] = {}
    channels: dict[int, set[str]] = defaultdict(set)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.id
            scores[chunk_id] += 1.0 / (rrf_k + rank)
            chunks[chunk_id] = item.chunk
            channels[chunk_id].add(item.channel)
    ranked = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
    return [ScoredChunk(chunks[i], scores[i], "+".join(sorted(channels[i]))) for i in ranked]


class HybridRetriever:
    def __init__(self, lexical: LexicalIndex, dense: DenseIndex | None = None, rrf_k: int = 60):
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 8, query_embedding: np.ndarray | None = None,
               allowed_ids: set[int] | None = None) -> list[ScoredChunk]:
        depth = max(top_k, top_k * 4)
        lexical = self.lexical.search(query, depth, allowed_ids)
        rankings: list[Sequence[ScoredChunk]] = [lexical]
        if self.dense is not None and query_embedding is not None:
            rankings.append(self.dense.search(query_embedding, depth, allowed_ids))
        if len(rankings) == 1:
            return lexical[:top_k]
        return rrf_fuse(rankings, top_k=top_k, rrf_k=self.rrf_k)

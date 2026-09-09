"""Synthetic multi-needle retrieval benchmark at a million model tokens.

The benchmark deliberately does not load model weights.  A local tokenizer is
used only to make the synthetic corpus length exact in model-token units.  It
places several documentation-like facts at controlled positions, adds draft
distractors, and measures whether the lexical retriever plus the adaptive
span selector can bring every fact into the bounded active window.

The same generated episodes can later be used for a model-generation pass:
the selected fragments are already the model input, while the expected values
are recorded in the JSON report.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from native_rag.documents import DocumentChunk  # noqa: E402
from native_rag.retrieval import LexicalIndex, ScoredChunk, select_adaptive_spans  # noqa: E402


DEFAULT_TOKENIZER = ROOT.parent / "Qwen3.5-0.8b"
DEFAULT_CORPUS_TOKENS = 1_000_000
DEFAULT_CHUNK_TOKENS = 128
DEFAULT_NEEDLES = 8

FILLER = (
    "The documentation describes ordinary operational guidance for a service. "
    "Teams review configuration changes, record observations, and preserve an "
    "audit trail before deploying a release. "
)

TOPICS = (
    ("archive snapshots", "retention period"),
    ("incident reports", "escalation window"),
    ("batch exports", "delivery schedule"),
    ("access reviews", "approval interval"),
    ("replication checks", "verification threshold"),
    ("maintenance notices", "publication lead time"),
    ("quota alerts", "warning boundary"),
    ("recovery drills", "verification code"),
    ("audit bundles", "storage tier"),
    ("release manifests", "rollback window"),
    ("service handoffs", "on-call interval"),
    ("dependency scans", "remediation deadline"),
)


@dataclass(frozen=True)
class Needle:
    index: int
    record: str
    topic: str
    field: str
    answer: str
    zone: str
    start_token: int
    end_token: int
    target_chunks: tuple[int, ...]


@dataclass(frozen=True)
class Episode:
    token_ids: list[int]
    chunks: list[DocumentChunk]
    needles: list[Needle]
    question: str


def position_zones() -> tuple[str, ...]:
    return ("start", "early", "center_left", "center", "center_right", "late", "tail", "end")


def _needle_text(record: str, topic: str, field: str, answer: str) -> str:
    return (
        f"Canonical record {record} governs {topic}. The approved {field} is "
        f"{answer}. This value is authoritative for the current documentation release."
    )


def _draft_text(record: str, topic: str, field: str, answer: str) -> str:
    return (
        f"Draft note for {record}: a provisional {field} for {topic} was "
        f"{answer}. This proposal is not the canonical approved value."
    )


def _replace_ids(corpus: list[int], start: int, replacement: list[int], occupied: list[tuple[int, int]]) -> None:
    end = start + len(replacement)
    if start < 0 or end > len(corpus):
        raise ValueError("synthetic needle does not fit inside the requested corpus")
    if any(start < previous_end and end > previous_start for previous_start, previous_end in occupied):
        raise ValueError("synthetic needle placement overlaps another fact")
    corpus[start:end] = replacement
    occupied.append((start, end))


def _chunk_ids_for_span(span: DocumentChunk, chunk_tokens: int, chunk_count: int) -> set[int]:
    if span.end_char <= span.start_char:
        return set()
    first = max(0, span.start_char // chunk_tokens)
    last = max(first, (span.end_char - 1) // chunk_tokens)
    return set(range(first, min(last + 1, chunk_count)))


def build_episode(tokenizer, corpus_tokens: int, chunk_tokens: int, needle_count: int, seed: int) -> Episode:
    """Create one exact-length episode using tokenizer IDs, without loading weights."""
    if corpus_tokens <= 0:
        raise ValueError("corpus_tokens must be positive")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not 2 <= needle_count <= len(position_zones()):
        raise ValueError(f"needle_count must be in [2, {len(position_zones())}]")

    rng = random.Random(seed)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("tokenizer produced no filler tokens")
    repeats = (corpus_tokens + len(filler_ids) - 1) // len(filler_ids)
    token_ids = (filler_ids * repeats)[:corpus_tokens]
    chunk_count = (corpus_tokens + chunk_tokens - 1) // chunk_tokens

    selected_topics = list(TOPICS[:needle_count])
    zones = position_zones()[:needle_count]
    fractions = (0.015, 0.12, 0.27, 0.49, 0.66, 0.80, 0.925, 0.975)
    offsets = (4, 24, 56, chunk_tokens - 14, 12, 40, chunk_tokens - 18, 32)
    needles: list[Needle] = []
    occupied: list[tuple[int, int]] = []

    for index, ((topic, field), zone, fraction, offset) in enumerate(
        zip(selected_topics, zones, fractions, offsets)
    ):
        record = f"registry-{index + 1:02d}-{rng.randrange(1000, 10000)}"
        answer = f"{rng.randrange(100000, 1000000)}"
        text = _needle_text(record, topic, field, answer)
        ids = tokenizer.encode(text, add_special_tokens=False)
        chunk_index = min(chunk_count - 2, max(1, int(chunk_count * fraction)))
        start = chunk_index * chunk_tokens + min(offset, chunk_tokens - 1)
        _replace_ids(token_ids, start, ids, occupied)
        end = start + len(ids)
        target_chunks = tuple(range(start // chunk_tokens, (end - 1) // chunk_tokens + 1))
        needles.append(Needle(
            index=index,
            record=record,
            topic=topic,
            field=field,
            answer=answer,
            zone=zone,
            start_token=start,
            end_token=end,
            target_chunks=target_chunks,
        ))

    # Add one far-away draft distractor per needle.  It shares the record and
    # topic vocabulary, but explicitly says that its value is not canonical.
    # This prevents the test from degenerating into a pure unique-ID lookup.
    for index, needle in enumerate(needles):
        draft_answer = f"{rng.randrange(100000, 1000000)}"
        ids = tokenizer.encode(
            _draft_text(needle.record, needle.topic, needle.field, draft_answer),
            add_special_tokens=False,
        )
        candidates = [
            ((index + 3) * chunk_count // (needle_count + 3) + 7 * index) % max(2, chunk_count - 2)
            for _ in range(chunk_count)
        ]
        for candidate in candidates:
            start = max(1, candidate * chunk_tokens)
            end = start + len(ids)
            if end <= corpus_tokens and not any(
                start < previous_end and end > previous_start
                for previous_start, previous_end in occupied
            ):
                _replace_ids(token_ids, start, ids, occupied)
                break
        else:
            raise ValueError("could not place all synthetic distractors")

    chunks: list[DocumentChunk] = []
    for chunk_id, start in enumerate(range(0, corpus_tokens, chunk_tokens)):
        ids = token_ids[start:start + chunk_tokens]
        chunks.append(DocumentChunk(
            id=chunk_id,
            source="synthetic-1m.md",
            text=tokenizer.decode(ids, skip_special_tokens=True),
            start_char=start,
            end_char=start + len(ids),
            heading="Synthetic documentation",
        ))

    question = (
        "Find the canonical approved value for every registry record below. "
        "Ignore draft and provisional notes. Return the values in record order. "
        + " ".join(f"{needle.record} concerns {needle.topic};" for needle in needles)
    )
    return Episode(token_ids, chunks, needles, question)


def evaluate_episode(
    episode: Episode,
    budget_tokens: int,
    candidate_k: int,
    neighbor_radius: int,
    chunk_tokens: int,
    span_size: int,
) -> dict:
    if budget_tokens <= 0 or budget_tokens % chunk_tokens:
        raise ValueError(f"budget_tokens must be a positive multiple of {chunk_tokens}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if neighbor_radius < 0:
        raise ValueError("neighbor_radius must be non-negative")
    budget_chunks = budget_tokens // chunk_tokens
    index = LexicalIndex(episode.chunks)
    ranked = index.search(episode.question, top_k=candidate_k)
    selected = select_adaptive_spans(
        ranked,
        episode.chunks,
        span_size=span_size,
        radius=neighbor_radius,
        max_small_chunks=budget_chunks,
    )
    ranked_ids = {item.chunk.id for item in ranked}
    selected_ids = set().union(*(
        _chunk_ids_for_span(item.chunk, chunk_tokens, len(episode.chunks))
        for item in selected
    )) if selected else set()

    per_needle = []
    for needle in episode.needles:
        seed_overlap = ranked_ids.intersection(needle.target_chunks)
        active_overlap = selected_ids.intersection(needle.target_chunks)
        seed_rank = next((rank for rank, item in enumerate(ranked, start=1)
                          if item.chunk.id in needle.target_chunks), None)
        per_needle.append({
            **asdict(needle),
            "seed_any_chunk_hit": bool(seed_overlap),
            "active_any_chunk_hit": bool(active_overlap),
            "seed_full_needle_hit": seed_overlap == set(needle.target_chunks),
            "active_full_needle_hit": active_overlap == set(needle.target_chunks),
            "seed_rank": seed_rank,
            "seed_chunks_hit": len(seed_overlap),
            "active_chunks_hit": len(active_overlap),
        })

    target_chunks = set().union(*(needle.target_chunks for needle in episode.needles))
    active_target_chunks = target_chunks.intersection(selected_ids)
    seed_any = sum(item["seed_any_chunk_hit"] for item in per_needle)
    active_any = sum(item["active_any_chunk_hit"] for item in per_needle)
    seed_full = sum(item["seed_full_needle_hit"] for item in per_needle)
    active_full = sum(item["active_full_needle_hit"] for item in per_needle)
    ranks = [item["seed_rank"] for item in per_needle if item["seed_rank"] is not None]
    by_zone = {
        zone: {
            "needles": sum(item["zone"] == zone for item in per_needle),
            "active_hits": sum(item["zone"] == zone and item["active_any_chunk_hit"] for item in per_needle),
            "active_full_hits": sum(item["zone"] == zone and item["active_full_needle_hit"] for item in per_needle),
        }
        for zone in position_zones()[:len(episode.needles)]
    }
    return {
        "source_tokens": len(episode.token_ids),
        "chunk_count": len(episode.chunks),
        "budget_tokens": budget_tokens,
        "budget_chunks": budget_chunks,
        "chunk_tokens": chunk_tokens,
        "span_size": span_size,
        "candidate_k": candidate_k,
        "neighbor_radius": neighbor_radius,
        "selected_span_count": len(selected),
        "selected_chunk_count": len(selected_ids),
        "selected_target_chunk_count": len(active_target_chunks),
        "target_chunk_precision": len(active_target_chunks) / max(1, len(selected_ids)),
        "needle_count": len(episode.needles),
        "seed_any_needle_recall": seed_any / len(episode.needles),
        "active_any_needle_recall": active_any / len(episode.needles),
        "seed_full_needle_recall": seed_full / len(episode.needles),
        "active_full_needle_recall": active_full / len(episode.needles),
        "all_needles_active": active_full == len(episode.needles),
        "mean_seed_rank": sum(ranks) / len(ranks) if ranks else None,
        "worst_seed_rank": max(ranks) if ranks else None,
        "position_recall": by_zone,
        "selected_small_ids": sorted(selected_ids),
        "per_needle": per_needle,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus-tokens", type=int, default=DEFAULT_CORPUS_TOKENS)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--needles", type=int, default=DEFAULT_NEEDLES)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-tokens", type=int, default=8_192)
    parser.add_argument("--candidate-k", type=int, default=128)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--span-size", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "multi_needle_1m_retrieval.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    print(
        f"Multi-needle retrieval-only: corpus={args.corpus_tokens} tokenizer-tokens "
        f"samples={args.samples} needles={args.needles} budget={args.budget_tokens} "
        f"small_chunk={args.chunk_tokens} span_size={args.span_size} "
        f"candidates={args.candidate_k} radius={args.neighbor_radius} "
        "weights_loaded=False lexical=BM25 CPU",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    results: list[dict] = []
    started = time.perf_counter()
    for sample_index in range(args.samples):
        build_started = time.perf_counter()
        episode = build_episode(
            tokenizer,
            args.corpus_tokens,
            args.chunk_tokens,
            args.needles,
            args.seed + sample_index,
        )
        result = evaluate_episode(
            episode,
            args.budget_tokens,
            args.candidate_k,
            args.neighbor_radius,
            args.chunk_tokens,
            args.span_size,
        )
        result["sample"] = sample_index + 1
        result["build_seconds"] = time.perf_counter() - build_started
        results.append(result)
        print(
            f"[{sample_index + 1}/{args.samples}] "
            f"seed_any={sum(item['seed_any_chunk_hit'] for item in result['per_needle'])}/"
            f"{args.needles} active_any={sum(item['active_any_chunk_hit'] for item in result['per_needle'])}/"
            f"{args.needles} active_full={sum(item['active_full_needle_hit'] for item in result['per_needle'])}/"
            f"{args.needles} selected={result['selected_chunk_count']} "
            f"build={result['build_seconds']:.2f}s",
            flush=True,
        )

    count = len(results)
    summary = {
        "benchmark": "Native RAG synthetic multi-needle retrieval-only",
        "corpus_tokens": args.corpus_tokens,
        "samples": count,
        "needles": args.needles,
        "budget_tokens": args.budget_tokens,
        "chunk_tokens": args.chunk_tokens,
        "span_size": args.span_size,
        "candidate_k": args.candidate_k,
        "neighbor_radius": args.neighbor_radius,
        "retriever": "LexicalIndex/BM25 CPU",
        "model_weights_loaded": False,
        "tokenizer": str(args.tokenizer),
        "mean_seed_any_needle_recall": sum(item["seed_any_needle_recall"] for item in results) / count,
        "mean_active_any_needle_recall": sum(item["active_any_needle_recall"] for item in results) / count,
        "mean_seed_full_needle_recall": sum(item["seed_full_needle_recall"] for item in results) / count,
        "mean_active_full_needle_recall": sum(item["active_full_needle_recall"] for item in results) / count,
        "all_needles_active_samples": sum(item["all_needles_active"] for item in results),
        "mean_target_chunk_precision": sum(item["target_chunk_precision"] for item in results) / count,
        "mean_selected_chunks": sum(item["selected_chunk_count"] for item in results) / count,
        "mean_selected_spans": sum(item["selected_span_count"] for item in results) / count,
        "mean_seed_rank": sum(item["mean_seed_rank"] for item in results if item["mean_seed_rank"] is not None)
        / max(1, sum(item["mean_seed_rank"] is not None for item in results)),
        "elapsed_seconds": time.perf_counter() - started,
        "per_sample": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== RESULT ===", flush=True)
    print(f"active_any_needle_recall={summary['mean_active_any_needle_recall']:.3f}", flush=True)
    print(f"active_full_needle_recall={summary['mean_active_full_needle_recall']:.3f}", flush=True)
    print(f"all_needles_active_samples={summary['all_needles_active_samples']}/{count}", flush=True)
    print(f"target_chunk_precision={summary['mean_target_chunk_precision']:.3f}", flush=True)
    print(f"elapsed_seconds={summary['elapsed_seconds']:.2f}", flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()

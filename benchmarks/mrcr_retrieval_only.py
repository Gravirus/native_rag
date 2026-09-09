"""MRCR retrieval-only evaluation; does not load model weights or use CUDA."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from native_rag.documents import DocumentChunk  # noqa: E402
from native_rag.retrieval import LexicalIndex, select_adaptive_spans  # noqa: E402


DEFAULT_CHUNK_TOKENS = 256
DEFAULT_DATA = Path(r"C:\Users\GrAvIRus\Desktop\4\mrcr_data")
DEFAULT_BIN_CACHE = ROOT.parent / "native_rag_legacy" / "cache" / "mrcr_bins.csv"


def longest_consecutive_run(values: set[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    best = current = 1
    for previous, value in zip(ordered, ordered[1:]):
        current = current + 1 if value == previous + 1 else 1
        best = max(best, current)
    return best


def count_ranges(values: set[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return 1 + sum(value != previous + 1 for previous, value in zip(ordered, ordered[1:]))


def small_ids_for_span(span: DocumentChunk, chunk_tokens: int, chunk_count: int) -> set[int]:
    """Translate an adaptive output span's token offsets back to small chunks."""
    if span.end_char <= span.start_char:
        return set()
    first = max(0, span.start_char // chunk_tokens)
    last = max(first, (span.end_char - 1) // chunk_tokens)
    return set(range(first, min(last + 1, chunk_count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT.parent / "Qwen3.5-0.8b")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bin-cache", type=Path, default=DEFAULT_BIN_CACHE)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-tokens", type=int, default=8_192)
    parser.add_argument("--chunk-tokens", type=int, choices=[128, 256], default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument(
        "--span-size",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="number of adjacent retrieval chunks combined into one context span",
    )
    parser.add_argument("--candidate-k", "--seed-chunks", dest="candidate_k", type=int, default=64)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "mrcr_retrieval_256k_b8192_neighbors.json",
    )
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict]:
    files = sorted(args.data.glob("8needle_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no MRCR parquet files found in {args.data}")
    rows = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    bins = pd.read_csv(args.bin_cache, index_col=0)
    if len(rows) != len(bins):
        raise ValueError("MRCR parquet rows and token-bin cache have different lengths")
    rows["n_tokens_o200k"] = bins["n_tokens_o200k"].to_numpy()
    rows = rows[
        (rows["n_needles"] == 8)
        & (rows["n_tokens_o200k"] > 131_072)
        & (rows["n_tokens_o200k"] <= 262_144)
    ]
    rng = random.Random(args.seed)
    if args.samples <= 0 or args.samples > len(rows):
        raise ValueError(f"samples must be in [1, {len(rows)}]")
    return [rows.iloc[index].to_dict() for index in rng.sample(range(len(rows)), args.samples)]


def evaluate_sample(
    tokenizer,
    row: dict,
    budget_chunks: int,
    candidate_k: int,
    neighbor_radius: int,
    chunk_tokens: int,
    span_size: int,
) -> dict:
    messages = json.loads(row["prompt"])
    question = messages[-1]["content"]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)

    answer = str(row["answer"])
    prefix = str(row["random_string_to_prepend"])
    answer_body = answer.removeprefix(prefix)
    needle_char = rendered.find(answer_body[:80])
    question_char = rendered.rfind(question)
    if needle_char < 0 or question_char < 0:
        raise ValueError("could not locate MRCR target or final question")
    needle_start = len(tokenizer.encode(rendered[:needle_char], add_special_tokens=False))
    needle_end = len(tokenizer.encode(rendered[:needle_char + len(answer_body)], add_special_tokens=False))
    question_start = len(tokenizer.encode(rendered[:question_char], add_special_tokens=False))
    chunks = [
        token_ids[start:start + chunk_tokens]
        for start in range(0, min(question_start, len(token_ids)), chunk_tokens)
    ]
    docs = [
        DocumentChunk(
            index,
            "mrcr",
            tokenizer.decode(chunk, skip_special_tokens=True),
            index * chunk_tokens,
            (index + 1) * chunk_tokens,
        )
        for index, chunk in enumerate(chunks)
    ]
    ranked = LexicalIndex(docs).search(question, top_k=candidate_k)
    expanded = select_adaptive_spans(
        ranked,
        docs,
        span_size=span_size,
        radius=neighbor_radius,
        max_small_chunks=budget_chunks,
    )
    ranked_ids = [item.chunk.id for item in ranked]
    expanded_small_ids = set().union(
        *(small_ids_for_span(item.chunk, chunk_tokens, len(chunks)) for item in expanded)
    ) if expanded else set()
    selected = expanded_small_ids
    first_chunk = needle_start // chunk_tokens
    last_chunk = max(first_chunk, (max(needle_end - 1, needle_start)) // chunk_tokens)
    small_target_chunks = set(range(first_chunk, min(last_chunk + 1, len(chunks))))
    target_chunks = small_target_chunks
    seed_overlap = set(ranked_ids).intersection(target_chunks)
    overlap = selected.intersection(target_chunks)
    seed_rank = ranked_ids.index(first_chunk) + 1 if first_chunk in ranked_ids else None
    expanded_rank = next(
        (index + 1 for index, item in enumerate(expanded)
         if first_chunk in small_ids_for_span(item.chunk, chunk_tokens, len(chunks))),
        None,
    )
    return {
        "source_tokens": len(token_ids),
        "chunk_count": len(chunks),
        "context_span_count": len(expanded),
        "needle_chunk": first_chunk,
        "needle_last_chunk": last_chunk,
        "needle_context_span": next(
            (index for index, item in enumerate(expanded)
             if first_chunk in small_ids_for_span(item.chunk, chunk_tokens, len(chunks))),
            None,
        ),
        "candidate_k": candidate_k,
        "chunk_tokens": chunk_tokens,
        "context_chunk_tokens": chunk_tokens * span_size,
        "span_size": span_size,
        "neighbor_radius": neighbor_radius,
        "needle_seed_rank": seed_rank,
        "needle_expanded_rank": expanded_rank,
        "seed_retrieval_hit": first_chunk in ranked_ids,
        "retrieval_hit": first_chunk in selected,
        "seed_any_target_chunk_hit": bool(seed_overlap),
        "any_target_chunk_hit": bool(overlap),
        "seed_target_chunks_hit": len(seed_overlap),
        "small_target_chunks": len(small_target_chunks),
        "target_chunks": len(target_chunks),
        "target_chunks_hit": len(overlap),
        "target_coverage": len(overlap) / max(1, len(target_chunks)),
        "target_contiguous_run": longest_consecutive_run(overlap),
        "target_contiguous_coverage": longest_consecutive_run(overlap) / max(1, len(target_chunks)),
        "selected_chunks": len(selected),
        "selected_ranges": count_ranges(selected),
        "context_precision": len(overlap) / max(1, len(selected)),
        "seed_ids": ranked_ids,
        "top_ids": [item.chunk.id for item in expanded],
        "selected_small_ids": sorted(selected),
    }


def main() -> None:
    args = parse_args()
    context_chunk_tokens = args.chunk_tokens * args.span_size
    if args.budget_tokens <= 0 or args.budget_tokens % args.chunk_tokens:
        raise ValueError(f"budget-tokens must be a positive multiple of {args.chunk_tokens}")
    if args.candidate_k <= 0:
        raise ValueError("candidate-k must be positive")
    if args.neighbor_radius < 0:
        raise ValueError("neighbor-radius must be non-negative")
    rows = load_rows(args)
    budget_chunks = args.budget_tokens // args.chunk_tokens
    print(
        f"MRCR retrieval-only: bin=256k samples={len(rows)} "
        f"budget={args.budget_tokens} ({budget_chunks} small chunks; "
        f"{args.budget_tokens // context_chunk_tokens} target spans) "
        f"small_chunk_tokens={args.chunk_tokens} span_size={args.span_size} "
        f"context_chunk_tokens={context_chunk_tokens} "
        f"candidates={args.candidate_k} "
        f"radius={args.neighbor_radius} lexical=BM25 CPU",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        result = evaluate_sample(
            tokenizer,
            row,
            budget_chunks,
            args.candidate_k,
            args.neighbor_radius,
            args.chunk_tokens,
            args.span_size,
        )
        results.append(result)
        print(
            f"[{index}/{len(rows)}] seed/expanded="
            f"{int(result['seed_retrieval_hit'])}/{int(result['retrieval_hit'])} "
            f"rank={result['needle_seed_rank'] or '-'} coverage={result['target_coverage']:.2f} "
            f"source_tokens={result['source_tokens']}",
            flush=True,
        )

    n = len(results)
    seed_hit_count = sum(item["seed_retrieval_hit"] for item in results)
    hit_count = sum(item["retrieval_hit"] for item in results)
    seed_any_hit_count = sum(item["seed_any_target_chunk_hit"] for item in results)
    any_hit_count = sum(item["any_target_chunk_hit"] for item in results)
    ranks = [item["needle_seed_rank"] for item in results if item["needle_seed_rank"] is not None]
    summary = {
        "benchmark": "MRCR v2 retrieval-only diagnostic",
        "bin": "256k",
        "samples": n,
        "budget_tokens": args.budget_tokens,
        "budget_chunks": budget_chunks,
        "chunk_tokens": args.chunk_tokens,
        "context_chunk_tokens": context_chunk_tokens,
        "span_size": args.span_size,
        "candidate_k": args.candidate_k,
        "neighbor_radius": args.neighbor_radius,
        "retriever": "LexicalIndex/BM25 CPU",
        "model_weights_loaded": False,
        "seed_retrieval_recall": seed_hit_count / n,
        "seed_retrieval_hits": seed_hit_count,
        "retrieval_recall": hit_count / n,
        "retrieval_hits": hit_count,
        "seed_any_target_chunk_recall": seed_any_hit_count / n,
        "seed_any_target_chunk_hits": seed_any_hit_count,
        "any_target_chunk_recall": any_hit_count / n,
        "any_target_chunk_hits": any_hit_count,
        "mean_target_coverage": sum(item["target_coverage"] for item in results) / n,
        "mean_target_contiguous_coverage": sum(item["target_contiguous_coverage"] for item in results) / n,
        "mean_selected_chunks": sum(item["selected_chunks"] for item in results) / n,
        "mean_selected_spans": sum(item["context_span_count"] for item in results) / n,
        "mean_selected_ranges": sum(item["selected_ranges"] for item in results) / n,
        "mean_context_precision": sum(item["context_precision"] for item in results) / n,
        "mean_hit_rank": sum(ranks) / len(ranks) if ranks else None,
        "elapsed_seconds": time.perf_counter() - started,
        "per_sample": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== RESULT ===", flush=True)
    print(f"seed_retrieval_recall={seed_hit_count}/{n} ({summary['seed_retrieval_recall']:.3f})", flush=True)
    print(f"expanded_retrieval_recall={hit_count}/{n} ({summary['retrieval_recall']:.3f})", flush=True)
    print(f"seed_any_target_chunk_recall={seed_any_hit_count}/{n} ({summary['seed_any_target_chunk_recall']:.3f})", flush=True)
    print(f"any_target_chunk_recall={any_hit_count}/{n} ({summary['any_target_chunk_recall']:.3f})", flush=True)
    print(f"mean_target_coverage={summary['mean_target_coverage']:.3f}", flush=True)
    print(f"mean_target_contiguous_coverage={summary['mean_target_contiguous_coverage']:.3f}", flush=True)
    print(f"mean_selected_chunks={summary['mean_selected_chunks']:.2f}", flush=True)
    print(f"mean_selected_spans={summary['mean_selected_spans']:.2f}", flush=True)
    print(f"mean_selected_ranges={summary['mean_selected_ranges']:.2f}", flush=True)
    print(f"mean_context_precision={summary['mean_context_precision']:.3f}", flush=True)
    print(f"mean_hit_rank={summary['mean_hit_rank']}", flush=True)
    print(f"elapsed_seconds={summary['elapsed_seconds']:.1f}", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()

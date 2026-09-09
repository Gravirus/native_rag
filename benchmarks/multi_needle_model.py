"""Run the synthetic multi-needle benchmark through a local instruct model.

The corpus is one million tokenizer tokens, but Native RAG passes only the
selected active window to the model. Retrieval diagnostics remain separate
from generation diagnostics so a missed needle is not confused with a model
that saw the needle but failed to copy its value.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from multi_needle_1m import build_episode, evaluate_episode  # noqa: E402
from native_rag.engine import NativeRAGEngine  # noqa: E402
from native_rag.model import load_backend  # noqa: E402
from native_rag.prompt import build_prompt_spec  # noqa: E402


DEFAULT_MODEL = ROOT.parent / "Qwen2.5-Coder"


def _ordered_answers(output: str, expected: list[str]) -> tuple[int, bool, list[str]]:
    """Return count and ordered-match status for six-digit expected values."""
    found = re.findall(r"(?<!\d)\d{6}(?!\d)", output)
    cursor = 0
    ordered = True
    hits = 0
    for answer in expected:
        try:
            cursor = output.index(answer, cursor) + len(answer)
        except ValueError:
            ordered = False
            continue
        hits += 1
    return hits, ordered and hits == len(expected), found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["auto", "generic", "qwen35"], default="qwen35")
    parser.add_argument("--dtype", choices=["nf4", "fp16", "fp32", "bf16"], default="nf4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefill-slice", type=int, default=128)
    parser.add_argument("--corpus-tokens", type=int, default=1_000_000)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--needles", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-tokens", type=int, default=8_192)
    parser.add_argument("--candidate-k", type=int, default=128)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--span-size", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "multi_needle_qwen25coder_nf4.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    print(
        f"Multi-needle model benchmark: model={args.model} backend={args.backend} dtype={args.dtype} "
        f"corpus={args.corpus_tokens} tokenizer-tokens samples={args.samples} "
        f"needles={args.needles} budget={args.budget_tokens} "
        f"small_chunk={args.chunk_tokens} span_size={args.span_size} "
        f"candidates={args.candidate_k} radius={args.neighbor_radius} "
        f"prefill_slice={args.prefill_slice} max_new={args.max_new_tokens}",
        flush=True,
    )
    if args.dtype == "nf4":
        print("INFO: loading 4-bit NF4 weights with FP16 compute.", flush=True)
    backend = load_backend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
    )
    engine = NativeRAGEngine(backend, prefill_slice=args.prefill_slice)
    results: list[dict] = []
    started = time.perf_counter()

    for sample_index in range(args.samples):
        episode_started = time.perf_counter()
        episode = build_episode(
            backend.tokenizer,
            args.corpus_tokens,
            args.chunk_tokens,
            args.needles,
            args.seed + sample_index,
        )
        retrieval = evaluate_episode(
            episode,
            args.budget_tokens,
            args.candidate_k,
            args.neighbor_radius,
            args.chunk_tokens,
            args.span_size,
        )
        # Re-run the deterministic selection to preserve scored spans for the
        # prompt builder. The retrieval-only result remains the source of
        # truth for recall metrics.
        from native_rag.retrieval import LexicalIndex, select_adaptive_spans

        ranked = LexicalIndex(episode.chunks).search(episode.question, top_k=args.candidate_k)
        selected = select_adaptive_spans(
            ranked,
            episode.chunks,
            span_size=args.span_size,
            radius=args.neighbor_radius,
            max_small_chunks=args.budget_tokens // args.chunk_tokens,
        )
        prompt = build_prompt_spec(episode.question, selected)
        generation = engine.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            thinking=args.thinking,
        )
        expected = [needle.answer for needle in episode.needles]
        answer_hits, ordered_match, found = _ordered_answers(generation.text, expected)
        result = {
            "sample": sample_index + 1,
            "retrieval": retrieval,
            "prompt_tokens": generation.prompt_tokens,
            "generated_tokens": len(generation.token_ids),
            "answer_hits": answer_hits,
            "answer_recall": answer_hits / len(expected),
            "ordered_exact": ordered_match,
            "expected_answers": expected,
            "found_six_digit_values": found,
            "output": generation.text,
            "elapsed_seconds": time.perf_counter() - episode_started,
        }
        results.append(result)
        active_any = retrieval["active_any_needle_recall"]
        print(
            f"[{sample_index + 1}/{args.samples}] "
            f"retrieval_active={active_any:.3f} "
            f"answers={answer_hits}/{len(expected)} ordered={int(ordered_match)} "
            f"prompt={generation.prompt_tokens} elapsed={result['elapsed_seconds']:.2f}s",
            flush=True,
        )

    count = len(results)
    summary = {
        "benchmark": "Native RAG synthetic multi-needle generation",
        "model": str(args.model),
        "backend": args.backend,
        "dtype": args.dtype,
        "corpus_tokens": args.corpus_tokens,
        "samples": count,
        "needles": args.needles,
        "budget_tokens": args.budget_tokens,
        "chunk_tokens": args.chunk_tokens,
        "span_size": args.span_size,
        "candidate_k": args.candidate_k,
        "neighbor_radius": args.neighbor_radius,
        "prefill_slice": args.prefill_slice,
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
        "mean_retrieval_active_recall": sum(item["retrieval"]["active_any_needle_recall"] for item in results) / count,
        "mean_retrieval_full_recall": sum(item["retrieval"]["active_full_needle_recall"] for item in results) / count,
        "mean_answer_recall": sum(item["answer_recall"] for item in results) / count,
        "ordered_exact_samples": sum(item["ordered_exact"] for item in results),
        "mean_prompt_tokens": sum(item["prompt_tokens"] for item in results) / count,
        "mean_elapsed_seconds": sum(item["elapsed_seconds"] for item in results) / count,
        "elapsed_seconds": time.perf_counter() - started,
        "per_sample": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== RESULT ===", flush=True)
    print(f"retrieval_active_recall={summary['mean_retrieval_active_recall']:.3f}", flush=True)
    print(f"retrieval_full_recall={summary['mean_retrieval_full_recall']:.3f}", flush=True)
    print(f"answer_recall={summary['mean_answer_recall']:.3f}", flush=True)
    print(f"ordered_exact_samples={summary['ordered_exact_samples']}/{count}", flush=True)
    print(f"mean_prompt_tokens={summary['mean_prompt_tokens']:.1f}", flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()

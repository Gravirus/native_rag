"""Direct long-context needle test for Qwen3.5, without retrieval or documents."""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from native_rag.engine import NativeRAGEngine  # noqa: E402
from native_rag.model import Qwen35Backend  # noqa: E402


NEEDLE = "The hidden ANSWER is ZEPHYR-731482."
EXPECTED = "731482"
FILLER = (
    "This is neutral background text for a long-context retrieval test. "
    "It contains no answer, no code, and no special instruction. "
)


def build_context(tokenizer, target_tokens: int, depth: float) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")

    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_ids = tokenizer.encode("\n\n" + NEEDLE + "\n\n", add_special_tokens=False)
    if len(needle_ids) >= target_tokens:
        raise ValueError("target_tokens is shorter than the needle")

    filler_count = target_tokens - len(needle_ids)
    repeats, remainder = divmod(filler_count, len(filler_ids))
    background = filler_ids * repeats + filler_ids[:remainder]
    split = int(len(background) * depth)
    token_ids = background[:split] + needle_ids + background[split:]
    return tokenizer.decode(token_ids, skip_special_tokens=False), split


def build_messages(context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating long-context recall. "
                "Use only the text between BEGIN and END. "
                "Answer with the six digits after the word ANSWER."
            ),
        },
        {
            "role": "user",
            "content": (
                "BEGIN LONG CONTEXT\n"
                f"{context}"
                "\nEND LONG CONTEXT\n\n"
                "Question: What is the six-digit answer hidden in the context? "
                "Reply with only those six digits."
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT.parent / "Qwen3.5-0.8b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["nf4", "fp16", "fp32", "bf16"], default="nf4")
    parser.add_argument("--length", type=int, default=260_000, help="approximate context tokens")
    parser.add_argument("--depth", type=float, default=0.5, help="needle location in [0, 1]")
    parser.add_argument("--prefill-slice", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dtype == "bf16":
        print("WARNING: this benchmark is intended to establish the FP32 baseline.", flush=True)
    if args.dtype == "nf4":
        print("INFO: running with 4-bit NF4 weights and FP16 compute.", flush=True)

    print("loading model", flush=True)
    backend = Qwen35Backend.load(args.model, device=args.device, dtype=args.dtype)
    context, requested_needle_position = build_context(backend.tokenizer, args.length, args.depth)
    messages = build_messages(context)
    rendered = backend.render_chat(messages, thinking=False)
    prompt_ids = backend.tokenizer.encode(rendered, add_special_tokens=False)
    if len(prompt_ids) + args.max_new_tokens >= backend.max_position_embeddings:
        raise ValueError(
            f"prompt ({len(prompt_ids)}) + generation ({args.max_new_tokens}) "
            f"reaches model limit {backend.max_position_embeddings}"
        )

    print(f"requested_context_tokens={args.length}", flush=True)
    print(f"actual_prompt_tokens={len(prompt_ids)}", flush=True)
    print(f"requested_needle_position={requested_needle_position}", flush=True)
    print(f"prefill_slice={args.prefill_slice}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    try:
        result = NativeRAGEngine(backend, args.prefill_slice).generate(
            messages, max_new_tokens=args.max_new_tokens, thinking=False
        )
    except RuntimeError:
        print("benchmark_status=failed", flush=True)
        raise
    elapsed = time.perf_counter() - started
    answer = result.text.strip()
    found = EXPECTED in re.sub(r"\s+", "", answer)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    reserved_gb = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"benchmark_status=ok", flush=True)
    print(f"elapsed_seconds={elapsed:.2f}", flush=True)
    print(f"peak_cuda_allocated_gb={peak_gb:.2f}", flush=True)
    print(f"peak_cuda_reserved_gb={reserved_gb:.2f}", flush=True)
    print(f"answer={answer!r}", flush=True)
    print(f"needle_hit={int(found)}", flush=True)

    del result, messages, rendered, context, backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Measure Native RAG with a 250k-token corpus and bounded active windows."""

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


CHUNK_TOKENS = 256
FILLER = (
    "This is neutral documentation background. It describes a generic process "
    "without the answer, without code, and without a special marker. "
)
NEEDLE = "The archival marker is LANTERN-42. The hidden ANSWER is ZEPHYR-731482."
RETRIEVAL_QUERY = "archival marker LANTERN-42"
QUESTION = "What six-digit answer is associated with archival marker LANTERN-42? Reply with only six digits."
EXPECTED = "731482"


def make_corpus(tokenizer, token_count: int) -> tuple[list[list[int]], int, int]:
    """Build a token corpus and place the needle at a deterministic middle chunk."""
    needle_ids = tokenizer.encode("\n" + NEEDLE + "\n", add_special_tokens=False)
    if len(needle_ids) >= token_count:
        raise ValueError("corpus is shorter than the needle")
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    background_count = token_count - len(needle_ids)
    background = (filler_ids * ((background_count // len(filler_ids)) + 1))[:background_count]
    insert_at = (token_count // 2 // CHUNK_TOKENS) * CHUNK_TOKENS
    source = background[:insert_at] + needle_ids + background[insert_at:]
    chunks = [source[start:start + CHUNK_TOKENS] for start in range(0, len(source), CHUNK_TOKENS)]
    needle_chunk = insert_at // CHUNK_TOKENS
    return chunks, needle_chunk, len(source)


def lexical_select(tokenizer, chunks: list[list[int]], budget_chunks: int) -> tuple[list[int], int]:
    """Select chunks with a CPU-only lexical score; no embedding model is used."""
    query_ids = set(tokenizer.encode(RETRIEVAL_QUERY, add_special_tokens=False))
    scores = [(len(query_ids.intersection(chunk)), index) for index, chunk in enumerate(chunks)]
    ranked = sorted(range(len(chunks)), key=lambda index: (-scores[index][0], index))
    selected = sorted(ranked[:budget_chunks])
    return selected, max(scores[index][0] for index in selected)


def centered_select(chunk_count: int, needle_chunk: int, budget_chunks: int) -> tuple[list[int], int]:
    """Select a contiguous oracle window centered on the known needle chunk."""
    start = max(0, min(needle_chunk - budget_chunks // 2, chunk_count - budget_chunks))
    selected = list(range(start, min(start + budget_chunks, chunk_count)))
    return selected, 0


def messages_for(tokenizer, chunks: list[list[int]], selected: list[int]) -> list[dict[str, str]]:
    selected_text = "\n\n".join(
        f"[chunk {index}] {tokenizer.decode(chunks[index], skip_special_tokens=False)}"
        for index in selected
    )
    return [
        {
            "role": "system",
            "content": (
                "You answer from selected documentation fragments. "
                "Do not invent facts. Return only the requested six digits."
            ),
        },
        {
            "role": "user",
            "content": (
                "Selected documentation fragments from a larger corpus:\n"
                "<documents>\n"
                f"{selected_text}\n"
                "</documents>\n\n"
                f"Question: {QUESTION}"
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT.parent / "Qwen3.5-0.8b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["nf4", "fp16", "fp32", "bf16"], default="nf4")
    parser.add_argument("--corpus-tokens", type=int, default=250_000)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2_000, 4_000, 16_000, 32_000])
    parser.add_argument(
        "--selection", choices=["lexical", "needle-centered"], default="lexical",
        help="lexical retrieval, or oracle-centered window to isolate model capacity",
    )
    parser.add_argument("--prefill-slice", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dtype == "nf4":
        print("INFO: 4-bit NF4 weights with FP16 compute.", flush=True)
    print("loading model", flush=True)
    backend = Qwen35Backend.load(args.model, device=args.device, dtype=args.dtype)
    chunks, needle_chunk, actual_corpus_tokens = make_corpus(backend.tokenizer, args.corpus_tokens)
    print(f"corpus_tokens={actual_corpus_tokens}", flush=True)
    print(f"corpus_chunks={len(chunks)} chunk_tokens={CHUNK_TOKENS}", flush=True)
    print(f"needle_chunk={needle_chunk}", flush=True)
    print(f"selection={args.selection}", flush=True)

    for budget_tokens in args.budgets:
        if budget_tokens <= 0 or budget_tokens % CHUNK_TOKENS:
            raise ValueError(f"budget must be a positive multiple of {CHUNK_TOKENS}: {budget_tokens}")
        budget_chunks = budget_tokens // CHUNK_TOKENS
        if args.selection == "lexical":
            selected, retrieval_score = lexical_select(backend.tokenizer, chunks, budget_chunks)
        else:
            selected, retrieval_score = centered_select(len(chunks), needle_chunk, budget_chunks)
        needle_active_offset = selected.index(needle_chunk) * CHUNK_TOKENS if needle_chunk in selected else -1
        selection_hit = needle_chunk in selected
        if needle_chunk not in selected:
            print(
                f"budget_tokens={budget_tokens} selection_hit=0 "
                f"selected_chunks={len(selected)} needle_active_offset={needle_active_offset} "
                f"retrieval_score={retrieval_score}",
                flush=True,
            )
            continue

        messages = messages_for(backend.tokenizer, chunks, selected)
        rendered = backend.render_chat(messages, thinking=False)
        prompt_ids = backend.tokenizer.encode(rendered, add_special_tokens=False)
        if len(prompt_ids) + args.max_new_tokens >= backend.max_position_embeddings:
            raise ValueError("active prompt plus generation exceeds model context")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        started = time.perf_counter()
        result = NativeRAGEngine(backend, args.prefill_slice).generate(
            messages, max_new_tokens=args.max_new_tokens, thinking=False
        )
        elapsed = time.perf_counter() - started
        answer = result.text.strip()
        hit = EXPECTED in re.sub(r"\s+", "", answer)
        allocated = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        reserved = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        print(
            f"budget_tokens={budget_tokens} budget_chunks={budget_chunks} "
            f"selected_chunks={len(selected)} active_source_tokens={len(selected) * CHUNK_TOKENS} "
            f"prompt_tokens={len(prompt_ids)} needle_active_offset={needle_active_offset} "
            f"selection_hit={int(selection_hit)} "
            f"needle_hit={int(hit)} retrieval_score={retrieval_score} "
            f"elapsed_seconds={elapsed:.2f} peak_allocated_gb={allocated:.2f} "
            f"peak_reserved_gb={reserved:.2f} answer={answer!r}",
            flush=True,
        )
        del result, messages, rendered
        gc.collect()


if __name__ == "__main__":
    main()

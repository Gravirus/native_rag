from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import RagSettings, default_device, default_model_path
from .engine import NativeRAGEngine
from .index import DocumentIndex
from .model import Qwen35Backend
from .prompt import build_messages


def _common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", choices=["nf4", "fp16", "fp32", "bf16"], default="nf4")
    parser.add_argument("--no-safe-kernels", action="store_true")


def _load_backend(args: argparse.Namespace) -> Qwen35Backend:
    if args.dtype == "bf16":
        print("WARNING: bf16 is experimental for Qwen3.5 GDN on this Windows/CUDA setup.", file=sys.stderr)
    if args.dtype == "nf4":
        print("INFO: loading 4-bit NF4 weights with FP16 compute.", file=sys.stderr)
    return Qwen35Backend.load(
        args.model, device=args.device, dtype=args.dtype, safe_kernels=not args.no_safe_kernels
    )


def cmd_index(args: argparse.Namespace) -> None:
    backend = _load_backend(args)
    index = DocumentIndex.build(
        args.docs, embedder=backend,
        chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars,
    )
    index.save(args.out)
    print(f"indexed {len(index.chunks)} chunks into {args.out}")


def cmd_ask(args: argparse.Namespace) -> None:
    index = DocumentIndex.load(args.index)
    backend = _load_backend(args)
    hits = index.search(
        args.question, top_k=args.top_k, embedder=backend, rrf_k=args.rrf_k,
        neighbor_radius=args.neighbor_radius, max_context_chunks=args.max_context_chunks,
        candidate_k=args.candidate_k,
        span_size=args.span_size,
    )
    messages = build_messages(args.question, hits)
    result = NativeRAGEngine(backend, args.prefill_slice).generate(
        messages, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, thinking=args.thinking,
    )
    print(result.text.strip())
    if args.show_sources:
        print("\nSources:")
        for hit in hits:
            print(f"- {hit.chunk.source} [{hit.channel}, score={hit.score:.5f}]")


def cmd_smoke(args: argparse.Namespace) -> None:
    backend = _load_backend(args)
    result = NativeRAGEngine(backend, args.prefill_slice).generate(
        [{"role": "user", "content": "Ответь одним словом: готов."}],
        max_new_tokens=args.max_new_tokens, thinking=False,
    )
    print(f"model={backend.model_path}")
    print(f"device={backend.device} dtype={backend.dtype}")
    print(f"max_position_embeddings={backend.max_position_embeddings}")
    print(f"prompt_tokens={result.prompt_tokens}")
    print(result.text.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="native-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="build a document index")
    index.add_argument("--docs", type=Path, required=True)
    index.add_argument("--out", type=Path, required=True)
    index.add_argument("--chunk-chars", type=int, default=1200)
    index.add_argument("--overlap-chars", type=int, default=200)
    _common_model_args(index)
    index.set_defaults(func=cmd_index)

    ask = sub.add_parser("ask", help="retrieve documentation and answer a question")
    ask.add_argument("--index", type=Path, required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--candidate-k", type=int, default=None,
                     help="candidate pool before span selection; defaults to 64 when neighbours are enabled")
    ask.add_argument("--span-size", type=int, default=1,
                     help="combine this many adjacent indexed chunks into one context span")
    ask.add_argument("--rrf-k", type=int, default=60)
    ask.add_argument("--prefill-slice", type=int, default=512)
    ask.add_argument("--max-new-tokens", type=int, default=512)
    ask.add_argument("--temperature", type=float, default=0.0)
    ask.add_argument("--thinking", action="store_true")
    ask.add_argument("--show-sources", action="store_true")
    ask.add_argument("--neighbor-radius", type=int, default=1,
                     help="include this many neighboring chunks per retrieved hit")
    ask.add_argument("--max-context-chunks", type=int, default=None)
    _common_model_args(ask)
    ask.set_defaults(func=cmd_ask)

    smoke = sub.add_parser("smoke", help="run a minimal model-only forward/decode")
    smoke.add_argument("--prefill-slice", type=int, default=512)
    smoke.add_argument("--max-new-tokens", type=int, default=16)
    _common_model_args(smoke)
    smoke.set_defaults(func=cmd_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

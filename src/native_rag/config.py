from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    configured = os.environ.get("NATIVE_RAG_MODEL")
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT.parent / "Qwen3.5-0.8b"


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass(frozen=True)
class RagSettings:
    model_path: Path = default_model_path()
    device: str = default_device()
    dtype: str = "nf4"
    chunk_chars: int = 1200
    overlap_chars: int = 200
    prefill_slice: int = 512
    max_new_tokens: int = 512
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.dtype not in {"nf4", "fp16", "fp32", "bf16"}:
            raise ValueError("dtype must be 'nf4', 'fp16', 'fp32' or 'bf16'")
        if self.chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        if not 0 <= self.overlap_chars < self.chunk_chars:
            raise ValueError("overlap_chars must be in [0, chunk_chars)")
        if self.prefill_slice <= 0:
            raise ValueError("prefill_slice must be positive")

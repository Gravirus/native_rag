from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from .prompt import PromptSpec


class BackendCapabilityError(RuntimeError):
    """Raised when an optional backend execution capability is unavailable."""


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities exposed by a concrete generator implementation.

    The first two flags describe the Native RAG execution path.  The dense
    feature flag is independent: a model can expose useful hidden-state
    features without exposing a safe incremental decode path, and vice versa.
    """

    supports_native_dense_features: bool = False
    supports_incremental_prefill: bool = False
    supports_kv_cache: bool = False

    @property
    def supports_native_execution(self) -> bool:
        return self.supports_incremental_prefill and self.supports_kv_cache


@dataclass(frozen=True)
class CompiledPrompt:
    """Model-local prompt representation produced after PromptSpec rendering."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    prompt_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int


@runtime_checkable
class NativeDecodeSession(Protocol):
    """Small model-neutral view of an active incremental decode session."""

    @property
    def next_logits(self) -> torch.Tensor:
        ...

    def step(self, token_id: int) -> torch.Tensor:
        """Consume one generated token and return logits for the next token."""
        ...


@runtime_checkable
class GeneratorBackend(Protocol):
    """Narrow waist between Native RAG and a language-model implementation."""

    model_path: Path
    device: str
    dtype: str

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    @property
    def max_position_embeddings(self) -> int:
        ...

    @property
    def eos_ids(self) -> set[int]:
        ...

    @property
    def dense_feature_descriptor(self) -> dict[str, Any] | None:
        ...

    def compile_prompt(
        self,
        prompt: PromptSpec | list[dict[str, str]],
        thinking: bool = False,
    ) -> CompiledPrompt:
        ...

    def generate(
        self,
        prompt: CompiledPrompt,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        thinking: bool = False,
    ) -> GenerationResult:
        ...

    def start_native_decode(
        self,
        prompt: CompiledPrompt,
        prefill_slice: int = 512,
    ) -> NativeDecodeSession:
        ...

    def decode(self, token_ids: Iterable[int]) -> str:
        ...

    def native_dense_features(self, texts: Iterable[str]) -> np.ndarray:
        ...

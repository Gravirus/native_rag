from __future__ import annotations

from typing import Any

import torch

from .backend import GeneratorBackend, GenerationResult


class NativeRAGEngine:
    """Coordinate RAG and generation without knowing model internals.

    A backend with incremental-prefill and KV-cache capabilities gets the
    Native RAG execution path. Other backends use their compatibility
    ``generate`` implementation; the retrieval and prompt contracts stay the
    same in both cases.
    """

    def __init__(self, backend: GeneratorBackend, prefill_slice: int = 512):
        if prefill_slice <= 0:
            raise ValueError("prefill_slice must be positive")
        self.backend = backend
        self.prefill_slice = prefill_slice

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> int:
        values = logits[0] if logits.ndim > 1 else logits
        if temperature == 0:
            return int(values.argmax().item())
        probabilities = torch.softmax(values / temperature, dim=-1)
        return int(torch.multinomial(probabilities, 1).item())

    def _generate_native(
        self,
        compiled_prompt,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        session = self.backend.start_native_decode(compiled_prompt, self.prefill_slice)
        eos_ids = self.backend.eos_ids
        generated: list[int] = []
        logits = session.next_logits
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                token = self._sample(logits, temperature)
                if token in eos_ids:
                    break
                generated.append(token)
                logits = session.step(token)
        return GenerationResult(
            text=self.backend.decode(generated),
            token_ids=generated,
            prompt_tokens=compiled_prompt.prompt_tokens,
        )

    def generate(
        self,
        prompt: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        thinking: bool = False,
    ) -> GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        compiled_prompt = self.backend.compile_prompt(prompt, thinking=thinking)
        total_tokens = compiled_prompt.prompt_tokens + max_new_tokens
        if total_tokens > self.backend.max_position_embeddings:
            raise ValueError(
                f"prompt plus generation has {total_tokens} tokens, "
                f"model limit is {self.backend.max_position_embeddings}"
            )

        capabilities = self.backend.capabilities
        if capabilities.supports_native_execution:
            return self._generate_native(compiled_prompt, max_new_tokens, temperature)
        return self.backend.generate(
            compiled_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            thinking=thinking,
        )

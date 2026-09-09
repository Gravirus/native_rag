from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from .model import Qwen35Backend


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int


class NativeRAGEngine:
    """Prefill the retrieved documentation once, then decode through DynamicCache."""

    def __init__(self, backend: Qwen35Backend, prefill_slice: int = 512):
        if prefill_slice <= 0:
            raise ValueError("prefill_slice must be positive")
        self.backend = backend
        self.prefill_slice = prefill_slice

    def _prefill(self, ids: torch.Tensor) -> tuple[torch.Tensor, DynamicCache, int]:
        cache = DynamicCache(config=self.backend.model.config)
        logits = None
        total = ids.shape[1]
        with torch.inference_mode():
            for start in range(0, total, self.prefill_slice):
                end = min(start + self.prefill_slice, total)
                part = ids[:, start:end].to(self.backend.device)
                positions = torch.arange(start, end, device=self.backend.device).unsqueeze(0)
                out = self.backend.model(
                    input_ids=part,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                logits = out.logits[:, -1, :].float()
        if logits is None:
            raise ValueError("cannot prefill an empty prompt")
        return logits, cache, total

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 512,
                 temperature: float = 0.0, thinking: bool = False) -> GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        rendered = self.backend.render_chat(messages, thinking=thinking)
        prompt_ids = self.backend.tokenizer.encode(rendered, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("chat template produced an empty prompt")
        if len(prompt_ids) >= self.backend.max_position_embeddings:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens, model limit is {self.backend.max_position_embeddings}"
            )
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        logits, cache, next_position = self._prefill(ids)
        generated: list[int] = []
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if temperature == 0:
                    token = int(logits[0].argmax())
                else:
                    probabilities = torch.softmax(logits[0] / temperature, dim=-1)
                    token = int(torch.multinomial(probabilities, 1))
                if token in self.backend.eos_ids:
                    break
                generated.append(token)
                token_tensor = torch.tensor([[token]], dtype=torch.long, device=self.backend.device)
                positions = torch.tensor([[next_position]], dtype=torch.long, device=self.backend.device)
                out = self.backend.model(
                    input_ids=token_tensor,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :].float()
                next_position += 1
        text = self.backend.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(text=text, token_ids=generated, prompt_tokens=len(prompt_ids))

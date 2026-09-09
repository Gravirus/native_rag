from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from .backend import BackendCapabilities, CompiledPrompt, NativeDecodeSession
from .model import (
    TransformersBackend,
    _load_transformers_model,
    _validate_load_args,
)


def _safe_torch_chunk_gated_delta_rule(module):
    """Return a CUDA-safe variant of Transformers' torch GDN fallback.

    The upstream fallback updates rows of ``attn`` and slices of the output
    buffer in-place. On the current Windows/PyTorch CUDA stack those writes can
    eventually surface as ``illegal memory access`` during long prefills. The
    math is unchanged; tensors are rebuilt functionally instead.
    """
    F = torch.nn.functional

    def safe_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            # Keep the upstream dtype for FP16/NF4. BF16 benefits from FP32
            # normalization before the rule is converted to its FP32 path.
            norm_dtype = torch.float32 if query.dtype == torch.bfloat16 else query.dtype
            query = module.l2norm(query.to(norm_dtype), dim=-1, eps=1e-6)
            key = module.l2norm(key.to(norm_dtype), dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        query = query * (1 / (query.shape[-1] ** 0.5))

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
        )
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        row_indices = torch.arange(chunk_size, device=attn.device)
        for i in range(1, chunk_size):
            row = attn[..., i, :i]
            sub = attn[..., :i, :i]
            updated = row + (row.unsqueeze(-1) * sub).sum(-2)
            updated_row = F.pad(updated, (0, chunk_size - i))
            update_mask = (row_indices == i).unsqueeze(-1) & (row_indices < i).unsqueeze(0)
            attn = torch.where(update_mask, updated_row.unsqueeze(-2), attn)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(
                batch_size, num_heads, k_head_dim, v_head_dim,
                dtype=value.dtype, device=value.device,
            )
            if initial_state is None
            else initial_state.to(value)
        )
        outputs = []
        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            local_attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            outputs.append(attn_inter + local_attn @ v_new)
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
                @ v_new
            )
        core_attn_out = torch.stack(outputs, dim=2)
        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    return safe_chunk_gated_delta_rule


def force_torch_linear_attention() -> bool:
    """Select the safe local GDN fallback before Qwen model construction."""
    try:
        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError:
        return False
    changed = False
    if hasattr(module, "torch_chunk_gated_delta_rule"):
        module.torch_chunk_gated_delta_rule = _safe_torch_chunk_gated_delta_rule(module)
        changed = True
    for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        if hasattr(module, name):
            setattr(module, name, None)
            changed = True
    return changed


class _QwenNativeDecodeSession:
    def __init__(self, backend: "Qwen35Backend", prompt: CompiledPrompt, prefill_slice: int):
        if prefill_slice <= 0:
            raise ValueError("prefill_slice must be positive")
        ids = prompt.input_ids
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("compiled prompt must contain one non-empty sequence")

        self.backend = backend
        self.cache = DynamicCache(config=backend.model.config)
        self._logits: torch.Tensor | None = None
        total = ids.shape[1]
        with torch.inference_mode():
            for start in range(0, total, prefill_slice):
                end = min(start + prefill_slice, total)
                part = ids[:, start:end].to(backend.device)
                positions = torch.arange(start, end, device=backend.device).unsqueeze(0)
                out = backend.model(
                    input_ids=part,
                    position_ids=positions,
                    past_key_values=self.cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                self._logits = out.logits[:, -1, :].float()
        if self._logits is None:
            raise ValueError("cannot prefill an empty prompt")
        self.next_position = total

    @property
    def next_logits(self) -> torch.Tensor:
        if self._logits is None:
            raise RuntimeError("decode session has no logits")
        return self._logits

    def step(self, token_id: int) -> torch.Tensor:
        token_tensor = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.backend.device)
        positions = torch.tensor([[self.next_position]], dtype=torch.long, device=self.backend.device)
        with torch.inference_mode():
            out = self.backend.model(
                input_ids=token_tensor,
                position_ids=positions,
                past_key_values=self.cache,
                use_cache=True,
            )
        self._logits = out.logits[:, -1, :].float()
        self.next_position += 1
        return self._logits


class Qwen35Backend(TransformersBackend):
    """Reference backend retaining the current Qwen-specific execution path."""

    def __init__(
        self,
        model,
        tokenizer,
        model_path: Path,
        device: str,
        dtype: str,
        embed_layer: int = 11,
    ):
        super().__init__(
            model,
            tokenizer,
            model_path,
            device,
            dtype,
            capabilities=BackendCapabilities(
                supports_native_dense_features=True,
                supports_incremental_prefill=True,
                supports_kv_cache=True,
            ),
        )
        self.embed_layer = int(embed_layer)

    @classmethod
    def load(
        cls,
        model_path: Path,
        device: str = "cuda",
        dtype: str = "nf4",
        safe_kernels: bool = True,
        embed_layer: int = 11,
    ) -> "Qwen35Backend":
        model_path = _validate_load_args(model_path, device, dtype)
        if safe_kernels:
            force_torch_linear_attention()
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        model_config = json.loads(model_path.joinpath("config.json").read_text(encoding="utf-8"))
        model_class = (
            AutoModelForCausalLM
            if model_config.get("model_type") == "qwen2"
            else AutoModelForImageTextToText
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = _load_transformers_model(model_class, model_path, device, dtype)
        return cls(model, tokenizer, model_path, device, dtype, embed_layer)

    @property
    def dense_feature_descriptor(self) -> dict[str, Any]:
        return {
            "provider": "model_native",
            "model_fingerprint": self.model_fingerprint,
            "architecture": str(getattr(self._text_config, "model_type", "qwen")),
            "feature_layer": self.embed_layer,
            "pooling": "mean",
            "normalization": "none",
            "dimension": self.hidden_size,
        }

    def _neutralize_document_text(self, text: str) -> str:
        # Preserve the exact transport behavior of the original Qwen backend,
        # then apply any additional tokenizer-specific escaping from the base.
        text = text.replace("<|im_start|>", "<im_start>").replace("<|im_end|>", "<im_end>")
        return super()._neutralize_document_text(text)

    def native_dense_features(self, texts: Iterable[str]) -> np.ndarray:
        """Mean-pool one hidden state per text using the loaded Qwen stack."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.hidden_size), dtype=np.float32)
        language_model = getattr(self.model.model, "language_model", self.model.model)
        layers = language_model.layers
        layer = layers[min(self.embed_layer, len(layers) - 1)]
        captured = {"value": None}

        def hook(_module, _args, output):
            captured["value"] = output[0] if isinstance(output, tuple) else output

        handle = layer.register_forward_hook(hook)
        vectors: list[np.ndarray] = []
        try:
            with torch.inference_mode():
                for text in texts:
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if not token_ids:
                        vectors.append(np.zeros(self.hidden_size, dtype=np.float32))
                        continue
                    ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                    positions = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
                    captured["value"] = None
                    self.model(input_ids=ids, position_ids=positions, use_cache=False)
                    if captured["value"] is None:
                        raise RuntimeError("Qwen hidden-state hook captured no output")
                    vectors.append(captured["value"][0].float().mean(dim=0).cpu().numpy())
        finally:
            handle.remove()
        return np.asarray(vectors, dtype=np.float32)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Backward-compatible alias for the model-native feature capability."""
        return self.native_dense_features(texts)

    def start_native_decode(
        self,
        prompt: CompiledPrompt,
        prefill_slice: int = 512,
    ) -> NativeDecodeSession:
        return _QwenNativeDecodeSession(self, prompt, prefill_slice)

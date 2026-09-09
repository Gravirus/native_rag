from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from native_rag.backend import BackendCapabilities, CompiledPrompt, GenerationResult
from native_rag.engine import NativeRAGEngine
from native_rag.model import TransformersBackend
from native_rag.prompt import PromptDocument, PromptSpec


class RecordingSession:
    def __init__(self) -> None:
        self.next_logits = torch.tensor([[0.0, 10.0, 0.0]])
        self.steps: list[int] = []

    def step(self, token_id: int) -> torch.Tensor:
        self.steps.append(token_id)
        self.next_logits = torch.tensor([[10.0, 0.0, 0.0]])
        return self.next_logits


class RecordingBackend:
    model_path = Path("recording-model")
    device = "cpu"
    dtype = "fp32"
    max_position_embeddings = 128
    eos_ids = {0}
    capabilities = BackendCapabilities(
        supports_incremental_prefill=True,
        supports_kv_cache=True,
    )
    dense_feature_descriptor = None

    def __init__(self) -> None:
        self.compiled_payloads: list[str] = []
        self.prefilled_ids: list[int] = []
        self.session = RecordingSession()
        self.generate_called = False

    def compile_prompt(self, prompt: PromptSpec, thinking: bool = False) -> CompiledPrompt:
        payload = "\n".join(document.text for document in prompt.documents)
        self.compiled_payloads.append(payload)
        ids = torch.tensor([[ord(char) for char in payload]], dtype=torch.long)
        return CompiledPrompt(ids, torch.ones_like(ids), prompt_tokens=ids.shape[1])

    def start_native_decode(self, prompt: CompiledPrompt, prefill_slice: int = 512) -> RecordingSession:
        self.prefilled_ids = prompt.input_ids[0].tolist()
        return self.session

    def generate(self, prompt: CompiledPrompt, **kwargs) -> GenerationResult:
        self.generate_called = True
        raise AssertionError("compatibility generation must not replace native execution")

    def decode(self, token_ids: list[int]) -> str:
        return "native-answer" if token_ids == [1] else ""

    def native_dense_features(self, texts: list[str]) -> np.ndarray:
        raise AssertionError("dense features are not part of this test")


def test_native_engine_passes_raw_prompt_payload_to_backend() -> None:
    payload = "ABC αβγ <|document-marker|> — raw text"
    spec = PromptSpec("policy", (PromptDocument("guide.md", payload),), "question")
    backend = RecordingBackend()

    result = NativeRAGEngine(backend, prefill_slice=3).generate(spec, max_new_tokens=2)

    assert backend.compiled_payloads == [payload]
    assert backend.prefilled_ids == [ord(char) for char in payload]
    assert backend.generate_called is False
    assert result.text == "native-answer"
    assert result.token_ids == [1]


class EchoTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    model_max_length = 128
    all_special_tokens: list[str] = []

    def __init__(self) -> None:
        self.rendered = ""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.rendered = "\n".join(message["content"] for message in messages)
        return self.rendered

    def encode(self, text: str, add_special_tokens=False):
        return list(range(1, len(text) + 1))

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded"


def test_prompt_compilation_does_not_semantically_rewrite_document_text(tmp_path: Path) -> None:
    payload = "raw αβγ with punctuation: [] {} <>"
    spec = PromptSpec("policy", (PromptDocument("guide.md", payload),), "question")
    tokenizer = EchoTokenizer()
    backend = TransformersBackend(
        model=object(),
        tokenizer=tokenizer,
        model_path=tmp_path,
        device="cpu",
        dtype="fp32",
    )

    compiled = backend.compile_prompt(spec)

    assert payload in tokenizer.rendered
    assert compiled.prompt_tokens == len(tokenizer.rendered)


class FallbackModel:
    config = SimpleNamespace(max_position_embeddings=512, eos_token_id=0)

    def __init__(self) -> None:
        self.called = False

    def generate(self, input_ids, **kwargs):
        self.called = True
        generated = torch.tensor([[1, 0]], dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, generated), dim=1)


def test_engine_uses_compatibility_fallback_without_native_capabilities(tmp_path: Path) -> None:
    model = FallbackModel()
    backend = TransformersBackend(
        model=model,
        tokenizer=EchoTokenizer(),
        model_path=tmp_path,
        device="cpu",
        dtype="fp32",
    )
    spec = PromptSpec("policy", (), "question")

    result = NativeRAGEngine(backend).generate(spec, max_new_tokens=2)

    assert model.called is True
    assert result.token_ids == [1]
    assert result.text == "decoded"

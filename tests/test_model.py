import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35

from native_rag.model import _safe_torch_chunk_gated_delta_rule


def test_safe_gdn_fallback_matches_upstream_on_cpu() -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 128, 2, 8)
    key = torch.randn(1, 128, 2, 8)
    value = torch.randn(1, 128, 2, 8)
    g = -torch.rand(1, 128, 2)
    beta = torch.rand(1, 128, 2)
    kwargs = {"chunk_size": 64, "output_final_state": True, "use_qk_l2norm_in_kernel": True}

    expected = qwen35.torch_chunk_gated_delta_rule(query, key, value, g, beta, **kwargs)
    actual = _safe_torch_chunk_gated_delta_rule(qwen35)(query, key, value, g, beta, **kwargs)

    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])

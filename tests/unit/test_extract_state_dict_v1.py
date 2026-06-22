"""Phase 5 unit tests for _extract_state_dict (v1)."""
import pytest
import torch

from src.models.classification.dinov3_base.base_classifier import _extract_state_dict


def test_extract_state_dict_mdcls_strips_net_prefix():
    ckpt = {
        "state_dict": {
            "net.feature.q_proj.weight": torch.zeros(64, 64),
            "net.classifier.weight": torch.zeros(4, 64),
            "net.classifier.bias": torch.zeros(4),
        },
        "epoch": 42,
    }
    out = _extract_state_dict(ckpt)
    assert set(out.keys()) == {"feature.q_proj.weight", "classifier.weight", "classifier.bias"}
    assert out["classifier.weight"].shape == (4, 64)


def test_extract_state_dict_sinchi_raw_raises_not_implemented():
    ckpt = {"model_state_dict": {"backbone.embed.weight": torch.zeros(1)}}
    with pytest.raises(NotImplementedError, match="Phase 8"):
        _extract_state_dict(ckpt)


def test_extract_state_dict_sinchi_peft_raises_not_implemented():
    ckpt = {"model_state_dict": {"backbone.base_model.model.q_proj.weight": torch.zeros(1)}}
    with pytest.raises(NotImplementedError, match="Phase 8"):
        _extract_state_dict(ckpt)


def test_extract_state_dict_unknown_format_raises_value_error():
    ckpt = {"random_key": 42, "another": "string"}
    with pytest.raises(ValueError, match="Unrecognised checkpoint format"):
        _extract_state_dict(ckpt)

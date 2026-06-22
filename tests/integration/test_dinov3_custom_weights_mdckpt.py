"""Phase 5 integration: load own MD-Cls ckpt via DINOv3CustomWeights."""
import pytest
import torch


def _build_mdcls_ckpt(num_classes: int = 4) -> dict:
    """Synthesize a Lightning-style .ckpt that DINOv3CustomWeights can load.

    The conftest stub backbone has 6 nn.Linear submodules (q_proj/k_proj/v_proj/
    o_proj/up_proj/down_proj) + pool (AdaptiveAvgPool2d). hidden_size=64.
    """
    g = torch.Generator().manual_seed(42)
    state = {}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"):
        state[f"net.feature.{proj}.weight"] = torch.empty(64, 64).normal_(generator=g)
        state[f"net.feature.{proj}.bias"] = torch.empty(64).normal_(generator=g)
    state["net.classifier.weight"] = torch.empty(num_classes, 64).normal_(generator=g)
    state["net.classifier.bias"] = torch.empty(num_classes).normal_(generator=g)
    return {"state_dict": state}


def test_dinov3_custom_weights_loads_mdcls_ckpt(mock_hf_automodel, tmp_path):
    from src.models.classification.dinov3_base import DINOv3CustomWeights

    ckpt_path = tmp_path / "dinov3_linear.ckpt"
    torch.save(_build_mdcls_ckpt(num_classes=4), ckpt_path)

    cw = DINOv3CustomWeights(
        weights=str(ckpt_path),
        class_names=["a", "b", "c", "d"],
        device="cpu",
        model_name="stub",
        embedding_dim=64,
    )
    assert cw.num_cls == 4
    assert torch.allclose(
        cw.net.classifier.bias,
        torch.load(ckpt_path, weights_only=False)["state_dict"]["net.classifier.bias"],
    )


def test_dinov3_custom_weights_forward(mock_hf_automodel, tmp_path):
    from src.models.classification.dinov3_base import DINOv3CustomWeights

    ckpt_path = tmp_path / "dinov3_linear.ckpt"
    torch.save(_build_mdcls_ckpt(num_classes=4), ckpt_path)
    cw = DINOv3CustomWeights(
        weights=str(ckpt_path),
        class_names=["a", "b", "c", "d"],
        device="cpu",
        model_name="stub",
        embedding_dim=64,
    )
    cw.eval()
    x = torch.randn(2, 3, 518, 518)
    with torch.no_grad():
        logits = cw(x)
    assert logits.shape == (2, 4)


def test_dinov3_custom_weights_missing_weights_raises(mock_hf_automodel):
    from src.models.classification.dinov3_base import DINOv3CustomWeights

    with pytest.raises(ValueError, match="weights path is required"):
        DINOv3CustomWeights(
            weights=None,
            class_names=["a"],
            device="cpu",
            model_name="stub",
            embedding_dim=64,
        )


def test_dinov3_custom_weights_corrupt_ckpt_raises(mock_hf_automodel, tmp_path):
    from src.models.classification.dinov3_base import DINOv3CustomWeights

    bad = tmp_path / "garbage.ckpt"
    bad.write_bytes(b"X" * 16)
    with pytest.raises(RuntimeError, match="Failed to torch.load"):
        DINOv3CustomWeights(
            weights=str(bad),
            class_names=["a"],
            device="cpu",
            model_name="stub",
            embedding_dim=64,
        )

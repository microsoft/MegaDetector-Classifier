"""Phase 4 integration tests for DINOv3Linear (linear probe)."""

import pytest
import pytorch_lightning as pl
import torch

from src.algorithms.dinov3_linear import DINOv3Linear


@pytest.fixture
def linear_probe_conf():
    return {
        "dinov3": {
            "model_name": "stub",
            "freeze_backbone": True,
            "peft": {"enable": False},
            "use_mixup": False,
        },
        "num_classes": 4,
        "lr_classifier": 1e-3,
    }


def test_dinov3_linear_instantiation(mock_hf_automodel, linear_probe_conf):
    model = DINOv3Linear(conf=linear_probe_conf)
    assert model.net.embedding_dim == 64
    assert model.net.classifier.out_features == 4


def test_dinov3_linear_forward(mock_hf_automodel, linear_probe_conf):
    pl.seed_everything(42, workers=True)
    model = DINOv3Linear(conf=linear_probe_conf)
    model.eval()
    x = torch.randn(2, 3, 518, 518)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 4)


def test_dinov3_linear_training_step_updates_classifier_only(mock_hf_automodel, linear_probe_conf):
    """One training_step should produce a loss; only classifier params should have grads."""
    pl.seed_everything(42, workers=True)
    model = DINOv3Linear(conf=linear_probe_conf)
    model.train()
    x = torch.randn(4, 3, 518, 518)
    y = torch.tensor([0, 1, 2, 3])
    loss = model.training_step((x, y), batch_idx=0)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.net.feature.named_parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)), (
            f"backbone param {name} got non-zero grad"
        )
    has_nonzero = False
    for name, p in model.net.classifier.named_parameters():
        assert p.grad is not None, f"classifier {name} got None grad"
        if (p.grad != 0).any():
            has_nonzero = True
    assert has_nonzero, "classifier received no gradient signal"


def test_dinov3_linear_peft_enable_raises(mock_hf_automodel):
    """conf.dinov3.peft.enable=True should raise NotImplementedError (Phase 7 stub)."""
    conf = {
        "dinov3": {
            "model_name": "stub",
            "freeze_backbone": True,
            "peft": {"enable": True},
            "use_mixup": False,
        },
        "num_classes": 4,
    }
    with pytest.raises(NotImplementedError, match="PEFT lands in Phase 7"):
        DINOv3Linear(conf=conf)


def test_dinov3_linear_full_ft_raises(mock_hf_automodel):
    """conf.dinov3.freeze_backbone=False should raise NotImplementedError (Phase 6 stub).

    This bubbles up from DINOv3Classifier's check.
    """
    conf = {
        "dinov3": {
            "model_name": "stub",
            "freeze_backbone": False,
            "peft": {"enable": False},
            "use_mixup": False,
        },
        "num_classes": 4,
    }
    with pytest.raises(NotImplementedError, match="Full FT lands in Phase 6"):
        DINOv3Linear(conf=conf)

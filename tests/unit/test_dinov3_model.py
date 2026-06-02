"""Phase 4 unit tests for DINOv3Classifier."""

import pytest
import torch

from src.models.dinov3 import DINOv3Classifier


def test_dinov3_classifier_instantiation(mock_hf_automodel):
    """Construct DINOv3Classifier with the conftest stub backbone."""
    model = DINOv3Classifier(model_name="stub", num_classes=4)
    assert model.embedding_dim == 64
    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.classifier.out_features == 4
    assert model.classifier.in_features == 64


def test_dinov3_classifier_freeze_backbone_true(mock_hf_automodel):
    """freeze_backbone=True freezes ALL backbone params."""
    model = DINOv3Classifier(model_name="stub", num_classes=4)
    for name, p in model.feature.named_parameters():
        assert p.requires_grad is False, f"{name} not frozen"
    for p in model.classifier.parameters():
        assert p.requires_grad is True


def test_dinov3_classifier_freeze_backbone_false_raises(mock_hf_automodel):
    """freeze_backbone=False raises NotImplementedError (Phase 6 stub)."""
    with pytest.raises(NotImplementedError, match="Full FT lands in Phase 6"):
        DINOv3Classifier(model_name="stub", num_classes=4, freeze_backbone=False)


def test_dinov3_classifier_explicit_embedding_dim(mock_hf_automodel):
    """Explicit embedding_dim overrides config.hidden_size."""
    model = DINOv3Classifier(model_name="stub", num_classes=4, embedding_dim=64)
    assert model.embedding_dim == 64


def test_dinov3_classifier_forward_shape(mock_hf_automodel):
    """Forward pass returns (B, num_classes)."""
    model = DINOv3Classifier(model_name="stub", num_classes=4)
    model.eval()
    x = torch.randn(2, 3, 518, 518)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 4), f"got {logits.shape}"

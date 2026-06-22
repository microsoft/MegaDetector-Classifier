import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.models.classification.dinov3_base.base_classifier import _extract_state_dict


def test_extract_state_dict_mdcls_strips_net_prefix():
    ckpt = {
        "state_dict": {
            "net.feature.q_proj.weight": torch.zeros(64, 64),
            "net.classifier.weight": torch.zeros(4, 64),
            "net.classifier.bias": torch.zeros(4),
        }
    }

    state, meta = _extract_state_dict(ckpt, return_meta=True)

    assert set(state) == {"feature.q_proj.weight", "classifier.weight", "classifier.bias"}
    assert meta["flavor"] == "md_cls"
    assert meta["num_classes"] == 4


def test_extract_state_dict_sinchi_raw_remaps_backbone_and_meta():
    ckpt = {
        "model_state_dict": {
            "backbone.embeddings.cls_token": torch.zeros(1, 1, 768),
            "backbone.layer.0.attention.q_proj.weight": torch.zeros(768, 768),
            "classifier.weight": torch.zeros(36, 768),
            "classifier.bias": torch.zeros(36),
        },
        "config": {
            "model": {
                "name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "num_classes": 36,
            },
            "data": {"image_size": 518},
            "peft": None,
        },
    }

    state, meta = _extract_state_dict(ckpt, return_meta=True)

    assert "feature.embeddings.cls_token" in state
    assert "feature.layer.0.attention.q_proj.weight" in state
    assert state["classifier.weight"].shape == (36, 768)
    assert "backbone.embeddings.cls_token" not in state
    assert meta == {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "num_classes": 36,
        "image_size": 518,
        "peft": None,
        "flavor": "sinchi_raw",
    }


def test_extract_state_dict_sinchi_peft_detects_remaps_and_meta():
    peft = {
        "method": "lora",
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ["q_proj"],
    }
    ckpt = {
        "model_state_dict": {
            "backbone.base_model.model.encoder.layer.0.q_proj.base_layer.weight": torch.zeros(1),
            "backbone.base_model.model.encoder.layer.0.q_proj.lora_A.default.weight": torch.zeros(1),
            "backbone.base_model.model.encoder.layer.0.q_proj.lora_B.default.weight": torch.zeros(1),
            "classifier.weight": torch.zeros(36, 384),
            "classifier.bias": torch.zeros(36),
        },
        "config": {
            "model": {
                "name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                "num_classes": 36,
            },
            "data": {"image_size": 518},
            "peft": peft,
        },
    }

    state, meta = _extract_state_dict(ckpt, return_meta=True)

    assert "feature.base_model.model.encoder.layer.0.q_proj.base_layer.weight" in state
    assert "feature.base_model.model.encoder.layer.0.q_proj.lora_A.default.weight" in state
    assert state["classifier.bias"].shape == (36,)
    assert meta["flavor"] == "sinchi_peft"
    assert meta["peft"] == peft
    assert meta["model_name"] == "facebook/dinov3-vits16plus-pretrain-lvd1689m"


def test_extract_state_dict_classifier_only_returns_head_and_meta():
    meta = {
        "format": "dinov3_classifier_only",
        "version": 1,
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "model_size": "base",
        "embedding_dim": 768,
        "num_classes": 36,
        "image_size": 518,
        "class_names": [str(i) for i in range(36)],
    }
    ckpt = {
        "classifier.weight": torch.zeros(36, 768),
        "classifier.bias": torch.zeros(36),
        "meta": meta,
    }

    state, out_meta = _extract_state_dict(ckpt, return_meta=True)

    assert set(state) == {"classifier.weight", "classifier.bias"}
    assert out_meta["model_name"] == meta["model_name"]
    assert out_meta["num_classes"] == 36
    assert out_meta["flavor"] == "classifier_only"


def test_extract_state_dict_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Top-level keys"):
        _extract_state_dict({"random_key": 42})


class _FakeDINOv3Classifier(nn.Module):
    def __init__(self, model_name, num_classes, embedding_dim=None, freeze_backbone=True):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim or 64
        self.feature = SimpleNamespace(config=SimpleNamespace(hidden_size=64))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(x)


def test_custom_weights_classifier_dim_mismatch_validation(monkeypatch):
    cw_mod = importlib.import_module("src.models.classification.dinov3_base.custom_weights")

    ckpt = {
        "classifier.weight": torch.zeros(2, 65),
        "classifier.bias": torch.zeros(2),
        "meta": {
            "model_name": "stub",
            "embedding_dim": 65,
            "num_classes": 2,
            "class_names": ["a", "b"],
        },
    }
    monkeypatch.setattr(cw_mod, "DINOv3Classifier", _FakeDINOv3Classifier)
    monkeypatch.setattr(cw_mod.torch, "load", lambda *args, **kwargs: ckpt)

    with pytest.raises(ValueError, match="classifier expects in_features=65"):
        cw_mod.DINOv3CustomWeights(
            weights="synthetic.pt",
            device="cpu",
            model_name="stub",
            embedding_dim=64,
        )

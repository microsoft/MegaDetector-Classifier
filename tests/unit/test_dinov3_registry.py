from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "models" / "dinov3_registry.py"
)


def _load_registry_module():
    spec = importlib.util.spec_from_file_location(
        "dinov3_registry_under_test", _REGISTRY_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REGISTRY = _load_registry_module()


def test_registry_has_required_sizes():
    assert (
        REGISTRY.DINOV3_MODEL_IDS["small_plus"]
        == "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    )
    assert (
        REGISTRY.DINOV3_MODEL_IDS["base"]
        == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    assert (
        REGISTRY.DINOV3_MODEL_IDS["huge_plus"]
        == "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    )

    assert REGISTRY.DINOV3_DIMS["small_plus"] == 384
    assert REGISTRY.DINOV3_DIMS["base"] == 768
    assert REGISTRY.DINOV3_DIMS["huge_plus"] == 1280


def test_resolve_model_id_accepts_size_or_full_hf_id():
    assert (
        REGISTRY.resolve_model_id("base")
        == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    assert REGISTRY.resolve_model_id("facebook/x/y") == "facebook/x/y"

    with pytest.raises(ValueError, match="Valid sizes"):
        REGISTRY.resolve_model_id("bogus")


def test_ensure_model_available_gated_repo_error_is_actionable(monkeypatch):
    class FakeGatedRepoError(Exception):
        pass

    class FakeRepositoryNotFoundError(Exception):
        pass

    class FakeLocalEntryNotFoundError(Exception):
        pass

    hub_module = types.ModuleType("huggingface_hub")
    hub_module.__path__ = []
    errors_module = types.ModuleType("huggingface_hub.errors")
    errors_module.GatedRepoError = FakeGatedRepoError
    errors_module.RepositoryNotFoundError = FakeRepositoryNotFoundError
    errors_module.LocalEntryNotFoundError = FakeLocalEntryNotFoundError

    def fake_try_to_load_from_cache(model_id, filename):
        assert model_id == REGISTRY.DINOV3_MODEL_IDS["base"]
        assert filename == "config.json"
        return None

    def fake_snapshot_download(*args, **kwargs):
        raise FakeGatedRepoError("license required")

    hub_module.errors = errors_module
    hub_module.try_to_load_from_cache = fake_try_to_load_from_cache
    hub_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)

    with pytest.raises(RuntimeError) as exc_info:
        REGISTRY.ensure_model_available(REGISTRY.DINOV3_MODEL_IDS["base"])

    message = str(exc_info.value)
    assert "accept the DINOv3 license" in message
    assert "hf auth login" in message
    assert "HF_TOKEN" in message

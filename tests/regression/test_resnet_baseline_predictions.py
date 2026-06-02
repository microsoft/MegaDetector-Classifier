from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torchvision.models.resnet as tvr

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover - compatibility with lightning>=2 installs
    import lightning.pytorch as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "baseline_predictions.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _seed_everything() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    try:
        pl.seed_everything(42, workers=True, verbose=False)
    except TypeError:
        pl.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(False)


def _fill_tensor_like(key: str, template: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if not torch.is_floating_point(template):
        return torch.zeros_like(template)

    result = torch.empty_like(template)
    if key.endswith("running_var"):
        return result.normal_(mean=1.0, std=0.02, generator=generator).abs_().add_(1e-3)
    if key.endswith("running_mean"):
        return result.normal_(mean=0.0, std=0.02, generator=generator)
    if key.endswith(".weight") and ("bn" in key or "downsample.1" in key):
        return result.normal_(mean=1.0, std=0.02, generator=generator)
    return result.normal_(mean=0.0, std=0.02, generator=generator)


def _make_synthetic_state_dict(arch: str) -> dict[str, torch.Tensor]:
    if arch == "resnet18":
        generator = torch.Generator().manual_seed(42)
        template = tvr.resnet18(weights=None).state_dict()
    elif arch == "resnet50":
        generator = torch.Generator().manual_seed(4242)
        template = tvr.resnet50(weights=None).state_dict()
    else:
        raise ValueError(f"Unsupported ResNet architecture: {arch}")
    return {key: _fill_tensor_like(key, value, generator) for key, value in template.items()}


def _make_input(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.empty(1, 3, 224, 224).normal_(generator=generator)


@pytest.fixture
def patched_resnet(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.plain_resnet as prn

    def _fake_load(url: str, *_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
        url_text = str(url)
        if "resnet18" in url_text:
            return _make_synthetic_state_dict("resnet18")
        if "resnet50" in url_text:
            return _make_synthetic_state_dict("resnet50")
        raise ValueError(f"Unexpected ResNet checkpoint URL: {url_text}")

    monkeypatch.setattr(prn, "load_state_dict_from_url", _fake_load)


@pytest.fixture(scope="module")
def golden() -> list[dict[str, int | float | str]]:
    with FIXTURE_PATH.open() as handle:
        return json.load(handle)


@pytest.mark.parametrize("entry_index", list(range(6)))
def test_resnet_baseline_prediction(patched_resnet: None, golden: list[dict[str, int | float | str]], entry_index: int) -> None:
    _seed_everything()
    entry = golden[entry_index]
    from src.models.plain_resnet import PlainResNetClassifier

    num_layers = 18 if entry["model"] == "resnet18" else 50
    model = PlainResNetClassifier(num_cls=2, num_layers=num_layers)
    model.feat_init()
    model.eval()
    x = _make_input(int(entry["input_seed"]))
    with torch.no_grad():
        feats = model.feature(x)
        logits = model.classifier(feats)
    actual_argmax = int(logits.argmax(dim=1).item())
    actual_conf = float(torch.softmax(logits, dim=1).max().item())

    assert actual_argmax == entry["argmax_label"], (
        f"{entry['model']} input_seed={entry['input_seed']}: "
        f"argmax {actual_argmax} != golden {entry['argmax_label']}"
    )
    assert abs(actual_conf - float(entry["confidence"])) <= 1e-4, (
        f"{entry['model']} input_seed={entry['input_seed']}: "
        f"confidence {actual_conf} not within 1e-4 of golden {entry['confidence']}"
    )

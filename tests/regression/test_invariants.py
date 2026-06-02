"""Phase 0 regression invariants I1, I2, I4-I12 (I3 lives in
test_resnet_baseline_predictions.py; I13 lands in Phase 5).

These invariants protect the ResNet classifier pipeline against
unintended changes during DINOv3 integration. See
docs/design/dinov3-integration/final.md §5.1 for the full table.
"""

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_RUNTIME_HASH = "f00b06e6ab206fa5f603305211a267267a11d71b153cb1e854a8a104a0001f3e"
EXPECTED_CONFIG = {
    "conf_id": "Crop_Res18_plain_071824",
    "algorithm": "Plain",
    "log_dir": "Crop",
    "num_epochs": 30,
    "log_interval": 10,
    "parallel": 0,
    "dataset_root": "./data/imgs",
    "dataset_name": "Custom_Crop",
    "annotation_dir": "./data/imgs",
    "split_path": "./data/imgs/annotation_example.csv",
    "test_size": 0.2,
    "val_size": 0.2,
    "split_data": True,
    "split_type": "location",
    "batch_size": 32,
    "num_workers": 4,
    "num_classes": 2,
    "model_name": "PlainResNetClassifier",
    "num_layers": 18,
    "weights_init": "ImageNet",
    "lr_feature": 0.01,
    "momentum_feature": 0.9,
    "weight_decay_feature": 0.0005,
    "lr_classifier": 0.01,
    "momentum_classifier": 0.9,
    "weight_decay_classifier": 0.0005,
    "step_size": 10,
    "gamma": 0.1,
}
EXPECTED_MAIN_SHA = "6c2b0225f04ae2527b34c9d41323465805ede6c221a240d0a94b7d397ec46dc7"
EXPECTED_PLAIN_SHA = "340fc47739932debf6a3ca1c406f8d8e21ee1b2cd0bb60a79b769ecb05638c2b"
EXPECTED_PLAIN_RESNET_SHA = "33a911f1f95f72d66e4f9999dcbaba3e8cf449f973096585f282cec4f0e7e276"
EXPECTED_CUSTOM_DATASET_SHA = "57a108435edd4889fa06a29dbc019050ab808e0dff4a22981b6185e2f40d65ee"


def _runtime_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_resnet50_checkpoint(num_cls: int = 2):
    from src.models.plain_resnet import PlainResNetClassifier

    model = PlainResNetClassifier(num_cls=num_cls, num_layers=50)
    state_dict = {f"net.{key}": value for key, value in model.state_dict().items()}
    return {"state_dict": state_dict}


@pytest.mark.isolated_env
def test_default_env_imports_clean():
    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--with-requirements",
            "requirements.txt",
            "python",
            "-c",
            "import src; print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "`import src` failed in default env.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_requirements_runtime_lines_hash():
    req = REPO_ROOT / "requirements.txt"
    lines = _runtime_lines(req.read_text())
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert digest == EXPECTED_RUNTIME_HASH, (
        "requirements.txt runtime deps changed.\n"
        f"Expected SHA-256: {EXPECTED_RUNTIME_HASH}\nActual: {digest}\n"
        f"Lines: {lines}"
    )


def test_default_config_yaml_equal():
    cfg_path = REPO_ROOT / "configs" / "config.yaml"
    with cfg_path.open() as f:
        actual = yaml.safe_load(f)
    assert actual == EXPECTED_CONFIG, (
        "configs/config.yaml dict differs from pre-change baseline.\n"
        f"Expected: {EXPECTED_CONFIG}\nActual: {actual}"
    )


@pytest.mark.isolated_env
def test_customweights_no_dinov3_deps():
    script = (
        "import src.models.classification.resnet_base.base_classifier as base_classifier\n"
        "from src.models.plain_resnet import PlainResNetClassifier\n"
        "from src.models.classification import CustomWeights\n"
        "model = PlainResNetClassifier(num_cls=2, num_layers=50)\n"
        "checkpoint = {'state_dict': {f'net.{k}': v for k, v in model.state_dict().items()}}\n"
        "base_classifier.torch.load = lambda *args, **kwargs: checkpoint\n"
        "cw = CustomWeights(weights='synthetic-resnet50.pt', class_names=['a', 'b'], device='cpu')\n"
        "assert cw.num_cls == 2\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--with-requirements",
            "requirements.txt",
            "python",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "CustomWeights construction/load failed in clean env:\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_plain_algorithm_class_identity():
    from src import algorithms

    assert "Plain" in algorithms.__dict__, "Plain missing from algorithms.__dict__"
    assert algorithms.__dict__["Plain"] is algorithms.Plain
    algorithms_reloaded = importlib.reload(algorithms)
    assert algorithms_reloaded.__dict__["Plain"] is algorithms_reloaded.Plain


def test_plain_resnet_class_identity():
    from src import models

    assert "PlainResNetClassifier" in models.__dict__
    assert models.__dict__["PlainResNetClassifier"] is models.PlainResNetClassifier


def test_main_unchanged():
    digest = _sha256_file(REPO_ROOT / "main.py")
    assert digest == EXPECTED_MAIN_SHA, (
        "main.py byte-equality violated. PERMANENT invariant.\n"
        f"Expected: {EXPECTED_MAIN_SHA}\nActual: {digest}"
    )


def test_plain_py_unchanged():
    digest = _sha256_file(REPO_ROOT / "src" / "algorithms" / "plain.py")
    assert digest == EXPECTED_PLAIN_SHA, (
        "src/algorithms/plain.py byte-equality violated.\n"
        f"Expected: {EXPECTED_PLAIN_SHA}\nActual: {digest}"
    )


def test_plain_resnet_py_unchanged():
    digest = _sha256_file(REPO_ROOT / "src" / "models" / "plain_resnet.py")
    assert digest == EXPECTED_PLAIN_RESNET_SHA, (
        "src/models/plain_resnet.py byte-equality violated.\n"
        f"Expected: {EXPECTED_PLAIN_RESNET_SHA}\nActual: {digest}"
    )


def test_custom_dataset_byte_or_behaviour():
    digest = _sha256_file(REPO_ROOT / "src" / "datasets" / "custom.py")
    assert digest == EXPECTED_CUSTOM_DATASET_SHA, (
        "src/datasets/custom.py byte-equality violated.\n"
        f"Expected: {EXPECTED_CUSTOM_DATASET_SHA}\nActual: {digest}"
    )


def test_customweights_legacy_call_signature(monkeypatch):
    import src.models.classification.resnet_base.base_classifier as base_classifier
    from src.models.classification import CustomWeights

    checkpoint = _synthetic_resnet50_checkpoint()
    monkeypatch.setattr(base_classifier.torch, "load", lambda *args, **kwargs: checkpoint)
    cw = CustomWeights(weights="synthetic-resnet50.pt", class_names=["a", "b"], device="cpu")
    assert cw.num_cls == 2

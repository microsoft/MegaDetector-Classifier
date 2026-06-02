from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _deterministic_tensor_like(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if value.is_floating_point():
        return torch.empty_like(value).normal_(generator=generator)
    return torch.zeros_like(value)


def _make_synth_ckpt(arch: str, num_cls: int = 2) -> dict[str, dict[str, torch.Tensor]]:
    """Build a synthetic Lightning-style .ckpt dict for CustomWeights to load."""
    import torchvision.models.resnet as tvr

    if arch == "resnet18":
        backbone = tvr.resnet18(weights=None)
        feat_dim = 512
        seed = 7
    elif arch == "resnet50":
        backbone = tvr.resnet50(weights=None)
        feat_dim = 2048
        seed = 70
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    generator = torch.Generator().manual_seed(seed)
    state_dict = {
        f"net.feature.{key}": _deterministic_tensor_like(value, generator)
        for key, value in backbone.state_dict().items()
    }
    state_dict["net.classifier.weight"] = torch.empty(num_cls, feat_dim).normal_(generator=generator)
    state_dict["net.classifier.bias"] = torch.empty(num_cls).normal_(generator=generator)
    return {"state_dict": state_dict}


def test_customweights_loads_resnet18(tmp_path: Path) -> None:
    from src.models.classification import CustomWeights

    ckpt_path = tmp_path / "synth_resnet18.ckpt"
    torch.save(_make_synth_ckpt("resnet18"), ckpt_path)
    cw = CustomWeights(weights=str(ckpt_path), class_names=["a", "b"], device="cpu", num_layers=18)
    assert cw.num_cls == 2


def test_customweights_loads_resnet50(tmp_path: Path) -> None:
    from src.models.classification import CustomWeights

    ckpt_path = tmp_path / "synth_resnet50.ckpt"
    torch.save(_make_synth_ckpt("resnet50"), ckpt_path)
    cw = CustomWeights(weights=str(ckpt_path), class_names=["a", "b"], device="cpu")
    assert cw.num_cls == 2

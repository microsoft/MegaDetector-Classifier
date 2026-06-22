import numpy as np
import pytest
import torch

from src.utils.mixup import efficient_batch_same_class_mixup, noisy_same_class_mixup


@pytest.fixture
def deterministic_rng(monkeypatch):
    def fixed_beta(a, b, size=None):
        if size is None:
            return 0.5
        return np.full(size, 0.5)

    def fixed_randint(low, high, size, **kwargs):
        return torch.zeros(size, dtype=torch.long, device=kwargs.get("device"))

    monkeypatch.setattr(np.random, "beta", fixed_beta)
    monkeypatch.setattr(torch, "randint", fixed_randint)


def test_output_tensor_shape_preserved(deterministic_rng):
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    x_mixed, y_mixed = noisy_same_class_mixup(x, y, mixup_alpha=0.4, noise_std=0.0)

    assert x_mixed.shape == x.shape
    assert y_mixed.shape == y.shape


def test_label_preservation_when_all_same_class(deterministic_rng):
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(6, 4)
    y = torch.full((6,), 7)

    _, y_mixed = noisy_same_class_mixup(x, y, mixup_alpha=0.4, noise_type="none")

    assert torch.equal(y_mixed, y)
    assert torch.unique(y_mixed).tolist() == [7]


def test_lambda_clamp_to_valid_range(monkeypatch):
    torch.manual_seed(42)
    np.random.seed(42)
    monkeypatch.setattr(np.random, "beta", lambda a, b, size=None: 1.5)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.zeros(
            size, dtype=torch.long, device=kwargs.get("device")
        ),
    )
    x = torch.tensor([[0.0], [10.0]])
    y = torch.tensor([3, 3])

    x_mixed, _ = noisy_same_class_mixup(x, y, mixup_alpha=0.4, noise_type="none")

    assert torch.allclose(x_mixed, torch.tensor([[1.0], [9.0]]), atol=1e-6)
    assert torch.all((x_mixed >= x.min()) & (x_mixed <= x.max()))


def test_singleton_class_edge_case(deterministic_rng):
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.tensor([[1.0, 2.0], [10.0, 20.0], [30.0, 40.0]])
    y = torch.tensor([0, 1, 1])

    x_mixed, y_mixed = noisy_same_class_mixup(x, y, mixup_alpha=0.4, noise_type="none")

    assert x_mixed.shape == x.shape
    assert torch.equal(y_mixed, y)
    assert torch.allclose(x_mixed[0], x[0])


def test_noise_type_modes_preserve_shape_and_dtype(deterministic_rng):
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(4, 5, dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1])

    for noise_type in ("gaussian", "dropout", "both"):
        x_mixed, y_mixed = noisy_same_class_mixup(
            x,
            y,
            mixup_alpha=0.4,
            noise_std=0.0,
            noise_type=noise_type,
            dropout_prob=0.0,
        )
        assert x_mixed.shape == x.shape
        assert x_mixed.dtype == x.dtype
        assert torch.equal(y_mixed, y)


def test_naive_and_efficient_implementations_are_equivalent(deterministic_rng, monkeypatch):
    torch.manual_seed(42)
    np.random.seed(42)

    def fixed_randperm(n, **kwargs):
        return torch.tensor([1, 0, 0, 0], dtype=torch.long, device=kwargs.get("device"))

    monkeypatch.setattr(torch, "randperm", fixed_randperm)
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    y = torch.zeros(4, dtype=torch.long)

    x_naive, y_naive = noisy_same_class_mixup(
        x, y, mixup_alpha=0.4, noise_type="none"
    )
    x_efficient, y_efficient = efficient_batch_same_class_mixup(
        x, y, mixup_alpha=0.4, noise_type="none"
    )

    assert torch.allclose(x_naive, x_efficient, atol=1e-6)
    assert torch.equal(y_naive, y_efficient)

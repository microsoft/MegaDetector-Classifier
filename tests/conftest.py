import torch
import torch.nn as nn
from types import SimpleNamespace
import pytest


class _StubConfig:
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size


class _StubBackbone(nn.Module):
    """Tiny ViT-shaped stub. Module names mirror HF DINOv3 ViT so PEFT
    target_modules tests are meaningful."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = _StubConfig(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, pixel_values=None, **kw):
        B = pixel_values.shape[0]
        h = self.pool(pixel_values.mean(1, keepdim=True)).flatten(1)
        h = h.expand(B, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=h.unsqueeze(1))


@pytest.fixture
def mock_hf_automodel(monkeypatch):
    def _from_pretrained(*a, **kw):
        return _StubBackbone(hidden_size=64)
    import transformers  # lazy import — fixture body only runs when test uses it
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained",
        staticmethod(_from_pretrained),
    )

from .resnet_base import *
from .timm_base import *
from .speciesnet_base import *
from .base_classifier import *

try:
    import importlib.util as _importlib_util

    for _dinov3_dep in ("transformers", "huggingface_hub"):
        if _importlib_util.find_spec(_dinov3_dep) is None:
            raise ModuleNotFoundError(
                f"No module named {_dinov3_dep!r}", name=_dinov3_dep
            )
    from .dinov3_base import *  # noqa: F401,F403
except ModuleNotFoundError as e:
    if e.name not in {"transformers", "huggingface_hub"}:
        raise

_DINOV3_NAMES = {
    "DINOv3AmazonRainforest",
    "DINOv3BaseInference",
    "DINOv3CustomWeights",
    "DINOv3Transform",
}


def __getattr__(name):
    if name in _DINOV3_NAMES:
        raise ImportError(
            f"{name} requires the optional DINOv3 dependencies. "
            f"Install with: pip install -r requirements-dinov3.txt"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

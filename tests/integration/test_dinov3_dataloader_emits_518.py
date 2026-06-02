"""Phase 3 integration: Custom_Crop(image_size=518) builds transforms emitting (B, 3, 518, 518) tensors."""

from PIL import Image

from src.datasets.custom import _build_data_transforms


def test_build_data_transforms_emits_518_for_dinov3():
    """_build_data_transforms(518) constructs transforms that output (3, 518, 518)."""
    t_dict = _build_data_transforms(518)
    img = Image.new("RGB", (640, 480), color=(127, 127, 127))

    out = t_dict['val'](img)
    assert out.shape == (3, 518, 518), (
        f"val transform output shape {out.shape} != (3, 518, 518)"
    )

    out_train = t_dict['train'](img)
    assert out_train.shape == (3, 518, 518), (
        f"train transform output shape {out_train.shape} != (3, 518, 518)"
    )

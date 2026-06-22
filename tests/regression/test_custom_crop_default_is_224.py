"""Phase 3 regression: Custom_Crop default image_size=224 reproduces prior transform sizes."""

from src.datasets.custom import data_transforms


def test_val_resize_is_224_by_default():
    """Default no-kwarg path: val transform should Resize to (224, 224)."""
    from torchvision.transforms import Resize

    val_transform = data_transforms['val']
    first_op = val_transform.transforms[0]
    assert isinstance(first_op, Resize)
    assert first_op.size == (224, 224), (
        f"data_transforms['val'] first op size = {first_op.size}, expected (224, 224)"
    )


def test_train_random_resized_crop_is_224_by_default():
    """Default no-kwarg path: train transform should RandomResizedCrop to (224, 224)."""
    from torchvision.transforms import RandomResizedCrop

    train_transform = data_transforms['train']
    first_op = train_transform.transforms[0]
    assert isinstance(first_op, RandomResizedCrop)
    assert first_op.size == (224, 224)

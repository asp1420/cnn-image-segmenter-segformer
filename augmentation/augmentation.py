import albumentations as aug
from albumentations import Compose


def train_transform(size: int) -> Compose:
    transforms = [
        aug.Resize(width=size, height=size),
        aug.OneOf([
            aug.HorizontalFlip(p=1.0),
            aug.VerticalFlip(p=1.0),
            aug.RandomRotate90(p=1.0),
        ], p=0.25),
        aug.GaussianBlur(blur_limit=(3, 5), p=0.15),
    ]
    transform = aug.Compose(transforms=transforms)
    return transform


def valid_transform(size: int) -> Compose:
    transform = aug.Compose(
        transforms=[
            aug.Resize(width=size, height=size)
        ]
    )
    return transform

"""PyTorch transforms for data augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import torchvision.transforms as transforms

from src.augmentation.methods import RandAugmentation, SequentialAugmentation
from src.augmentation.transforms import FILLCOLOR, SquarePad


def simple_augment_train(img_size: float = 32) -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),
            transforms.RandomResizedCrop(
                size=img_size, ratio=(0.75, 1.0, 1.3333333333333333)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ]
    )


def simple_augment_test(img_size: float = 32) -> transforms.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ]
    )


def randaugment_train(
    img_size: float = 32,
    n_select: int = 2,
    level: int = 14,
    n_level: int = 31,
) -> transforms.Compose:
    """Random augmentation policy for training CIFAR100."""
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 0.8, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ]
    )

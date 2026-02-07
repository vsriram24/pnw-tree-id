"""Data augmentation and transform pipelines for training and inference."""

from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 384
RESIZE_SIZE = 422  # ~10% larger for center crop


def get_train_transforms() -> transforms.Compose:
    """Training augmentation pipeline.

    Includes random resize crop, flips, color jitter, rotation,
    and random erasing for regularization.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation/test transform pipeline.

    Deterministic resize + center crop, no augmentation.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """Inference transforms (same as validation)."""
    return get_val_transforms()

"""EfficientNetV2-S model with custom classifier head for tree identification."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create EfficientNetV2-S with a custom classification head.

    Architecture:
        EfficientNetV2-S backbone (frozen or unfrozen)
        -> Dropout(0.3)
        -> Linear(1280, 512)
        -> ReLU
        -> BatchNorm1d(512)
        -> Dropout(0.15)
        -> Linear(512, num_classes)

    Args:
        num_classes: Number of tree species to classify.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Modified EfficientNetV2-S model.
    """
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_v2_s(weights=weights)

    # Get the input features of the original classifier
    in_features = model.classifier[1].in_features  # 1280

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.15),
        nn.Linear(512, num_classes),
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for fine-tuning."""
    for param in model.features.parameters():
        param.requires_grad = True


def get_parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list:
    """Create parameter groups with differential learning rates.

    Args:
        model: The EfficientNetV2-S model.
        backbone_lr: Learning rate for the backbone (features).
        head_lr: Learning rate for the classifier head.

    Returns:
        List of parameter group dicts for the optimizer.
    """
    return [
        {"params": model.features.parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ]

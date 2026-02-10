#!/usr/bin/env python3
"""CLI script to evaluate the trained model on the test set."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.dataset.transforms import get_val_transforms
from src.model.architecture import create_model
from src.training.metrics import (
    compute_classification_metrics,
    compute_topk_accuracy,
    plot_confusion_matrix,
)


def evaluate(
    checkpoint_path: str = "checkpoints/best_model.pt",
    data_dir: str = "data/processed/test",
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = "checkpoints",
) -> dict:
    """Evaluate the model on the test set.

    Returns dict with metrics.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = checkpoint["num_classes"]

    print(f"Model trained on {num_classes} classes")
    print(f"Best val_loss: {checkpoint.get('val_loss', 'N/A')}, "
          f"val_acc: {checkpoint.get('val_acc', 'N/A')}")

    # Load model
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = ImageFolder(data_dir, transform=get_val_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Test set: {len(test_dataset)} images, {len(test_dataset.classes)} classes")

    # Run inference
    all_preds = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_outputs.append(outputs.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.tensor(all_targets)

    # Compute metrics
    class_names = [
        idx_to_class.get(i, test_dataset.classes[i])
        for i in range(num_classes)
    ]

    metrics = compute_classification_metrics(all_preds, all_targets, class_names)
    top5_acc = compute_topk_accuracy(all_outputs, all_targets_tensor, k=5)

    print("\n=== Test Set Results ===")
    print(f"Top-1 Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"F1 (macro):     {metrics['overall']['f1_macro']:.4f}")
    print(f"F1 (weighted):  {metrics['overall']['f1_weighted']:.4f}")
    print(f"\n{metrics['report']}")

    # Save confusion matrix
    cm_path = f"{output_dir}/confusion_matrix.png"
    plot_confusion_matrix(all_preds, all_targets, class_names, save_path=cm_path)

    return {**metrics["overall"], "top5_accuracy": top5_acc}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained tree classifier on test set"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/test",
        help="Path to test data directory (default: data/processed/test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Directory to save confusion matrix (default: checkpoints)",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

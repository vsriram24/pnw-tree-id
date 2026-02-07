"""Training and evaluation metrics for tree classification."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)


def compute_topk_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> float:
    """Compute top-k accuracy."""
    _, predicted = outputs.topk(k, dim=1)
    correct = predicted.eq(targets.unsqueeze(1).expand_as(predicted)).any(1).sum().item()
    return correct / targets.size(0)


def compute_classification_metrics(
    all_preds: list[int],
    all_targets: list[int],
    class_names: list[str] | None = None,
) -> dict:
    """Compute per-class and overall classification metrics.

    Returns:
        Dict with 'report' (str), 'per_class' (dict), and 'overall' (dict).
    """
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )

    overall = {
        "accuracy": np.mean(np.array(all_preds) == np.array(all_targets)),
        "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            all_targets, all_preds, average="weighted", zero_division=0
        ),
        "precision_macro": precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            all_targets, all_preds, average="macro", zero_division=0
        ),
    }

    per_class = {}
    if class_names:
        p = precision_score(
            all_targets, all_preds, average=None, zero_division=0
        )
        r = recall_score(
            all_targets, all_preds, average=None, zero_division=0
        )
        f = f1_score(
            all_targets, all_preds, average=None, zero_division=0
        )
        for i, name in enumerate(class_names):
            if i < len(p):
                per_class[name] = {
                    "precision": float(p[i]),
                    "recall": float(r[i]),
                    "f1": float(f[i]),
                }

    return {"report": report, "overall": overall, "per_class": per_class}


def plot_confusion_matrix(
    all_preds: list[int],
    all_targets: list[int],
    class_names: list[str],
    save_path: str = "confusion_matrix.png",
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(all_targets, all_preds)
    fig_size = max(12, len(class_names) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=len(class_names) <= 20,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

"""Training loop with two-phase fine-tuning, early stopping, and checkpointing."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.dataset.transforms import get_train_transforms, get_val_transforms
from src.model.architecture import (
    create_model,
    freeze_backbone,
    get_parameter_groups,
    unfreeze_backbone,
)
from src.training.metrics import compute_accuracy


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="  Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="  Val", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    class_to_idx: dict,
    save_path: str,
) -> None:
    """Save model checkpoint with class mapping."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "class_to_idx": class_to_idx,
            "num_classes": len(class_to_idx),
        },
        save_path,
    )


def train(
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    phase1_epochs: int = 5,
    phase2_epochs: int = 30,
    batch_size: int = 32,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    fine_tune_lr: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 7,
    num_workers: int = 4,
) -> dict:
    """Run two-phase training.

    Phase 1: Freeze backbone, train head only.
    Phase 2: Unfreeze backbone, fine-tune with differential LR.

    Returns dict with training history.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = ImageFolder(
        f"{data_dir}/train", transform=get_train_transforms()
    )
    val_dataset = ImageFolder(
        f"{data_dir}/val", transform=get_val_transforms()
    )

    num_classes = len(train_dataset.classes)
    class_to_idx = train_dataset.class_to_idx
    print(f"Found {num_classes} classes, {len(train_dataset)} train images, "
          f"{len(val_dataset)} val images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    model = create_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history = {"phase1": [], "phase2": []}
    best_val_loss = float("inf")

    # ===== Phase 1: Train head only =====
    print("\n=== Phase 1: Training classifier head (backbone frozen) ===")
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=head_lr)

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch}/{phase1_epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )
        history["phase1"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                class_to_idx,
                f"{checkpoint_dir}/best_model.pt",
            )

    # ===== Phase 2: Fine-tune entire model =====
    print("\n=== Phase 2: Fine-tuning entire model ===")

    # Reduce batch size for phase 2 to avoid OOM on memory-constrained GPUs
    phase2_batch = max(batch_size // 2, 4)
    if phase2_batch != batch_size:
        print(f"  Reducing batch size to {phase2_batch} for full model fine-tuning")
        train_loader = DataLoader(
            train_dataset,
            batch_size=phase2_batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=phase2_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    unfreeze_backbone(model)
    param_groups = get_parameter_groups(model, backbone_lr, fine_tune_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-7)
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch}/{phase2_epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
            f"lr: {scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)"
        )
        history["phase2"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                class_to_idx,
                f"{checkpoint_dir}/best_model.pt",
            )
            print("    -> Saved best model")

        if early_stopping.step(val_loss):
            print(f"  Early stopping triggered after epoch {epoch}")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_loss, val_acc,
        class_to_idx,
        f"{checkpoint_dir}/final_model.pt",
    )

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history

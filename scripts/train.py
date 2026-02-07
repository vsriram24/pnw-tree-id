#!/usr/bin/env python3
"""CLI script to train the PNW tree identification model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Train PNW tree species classifier"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory with train/val/test splits (default: data/processed)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=5,
        help="Number of epochs for phase 1 (head only) (default: 5)",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=30,
        help="Max epochs for phase 2 (fine-tuning) (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience (default: 7)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    args = parser.parse_args()

    history = train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        num_workers=args.num_workers,
    )

    total_epochs = len(history["phase1"]) + len(history["phase2"])
    print(f"\nCompleted {total_epochs} total epochs")


if __name__ == "__main__":
    main()

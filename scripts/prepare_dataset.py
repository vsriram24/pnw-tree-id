#!/usr/bin/env python3
"""CLI script to preprocess and split the downloaded dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.preprocess import preprocess_all
from src.dataset.split import split_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and split PNW tree image dataset"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory with raw downloaded images (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for split dataset (default: data/processed)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=384,
        help="Target longest edge in pixels (default: 384)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (validate/resize/dedup), go straight to split",
    )
    args = parser.parse_args()

    if not args.skip_preprocess:
        print("=== Step 1: Preprocessing ===")
        preprocess_stats = preprocess_all(args.raw_dir, args.target_size)
        total_kept = sum(s["kept"] for s in preprocess_stats.values())
        print(f"Preprocessing complete: {total_kept} images kept\n")
    else:
        print("Skipping preprocessing step\n")

    print("=== Step 2: Splitting ===")
    split_stats = split_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )

    print("\n=== Split Summary ===")
    total_train = total_val = total_test = 0
    for slug, counts in sorted(split_stats.items()):
        t, v, te = counts["train"], counts["val"], counts["test"]
        total_train += t
        total_val += v
        total_test += te
        print(f"  {slug}: train={t}, val={v}, test={te}")

    print(f"\n  Total: train={total_train}, val={total_val}, test={total_test}")
    print(f"  Grand total: {total_train + total_val + total_test}")


if __name__ == "__main__":
    main()

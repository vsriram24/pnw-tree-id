#!/usr/bin/env python3
"""CLI script to download tree images from iNaturalist."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.download import download_all


def main():
    parser = argparse.ArgumentParser(
        description="Download PNW tree images from iNaturalist"
    )
    parser.add_argument(
        "--config",
        default="config/species.yaml",
        help="Path to species config YAML (default: config/species.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for downloaded images (default: data/raw)",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Download only specific species (by common name)",
    )
    args = parser.parse_args()

    results = download_all(
        config_path=args.config,
        output_dir=args.output_dir,
        species_filter=args.species,
    )

    print("\n=== Download Summary ===")
    total = 0
    for name, count in sorted(results.items()):
        print(f"  {name}: {count}")
        total += count
    print(f"  Total: {total} images across {len(results)} species")


if __name__ == "__main__":
    main()

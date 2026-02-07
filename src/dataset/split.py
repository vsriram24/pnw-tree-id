"""Stratified train/val/test split into ImageFolder-compatible layout."""

import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split preprocessed images into train/val/test directories.

    Creates ImageFolder-compatible layout:
        output_dir/train/{species_slug}/image.jpg
        output_dir/val/{species_slug}/image.jpg
        output_dir/test/{species_slug}/image.jpg

    Returns dict mapping species to split counts.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    # Create split directories
    for split in ("train", "val", "test"):
        (out_path / split).mkdir(parents=True, exist_ok=True)

    species_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    split_stats = {}

    for species_dir in species_dirs:
        slug = species_dir.name
        images = sorted(list(species_dir.glob("*.jpg")))

        if len(images) < 10:
            print(f"  Warning: {slug} has only {len(images)} images, skipping")
            continue

        # First split: train vs (val+test)
        train_imgs, valtest_imgs = train_test_split(
            images,
            train_size=train_ratio,
            random_state=seed,
        )

        # Second split: val vs test
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            valtest_imgs,
            train_size=relative_val,
            random_state=seed,
        )

        # Copy files to split directories
        splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
        counts = {}
        for split_name, split_imgs in splits.items():
            split_species_dir = out_path / split_name / slug
            split_species_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                dest = split_species_dir / img_path.name
                shutil.copy2(img_path, dest)
            counts[split_name] = len(split_imgs)

        split_stats[slug] = counts

    return split_stats

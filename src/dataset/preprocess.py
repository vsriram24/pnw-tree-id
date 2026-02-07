"""Validate, resize, and deduplicate downloaded images."""

import hashlib
from pathlib import Path

from PIL import Image
from tqdm import tqdm

TARGET_SIZE = 384


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(filepath: Path) -> bool:
    """Check that a file is a valid, openable image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def resize_image(filepath: Path, target_longest_edge: int = TARGET_SIZE) -> bool:
    """Resize image so longest edge equals target, maintaining aspect ratio.

    Overwrites the original file. Returns True on success.
    """
    try:
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) <= target_longest_edge:
                img.save(filepath, "JPEG", quality=95)
                return True
            if w >= h:
                new_w = target_longest_edge
                new_h = int(h * target_longest_edge / w)
            else:
                new_h = target_longest_edge
                new_w = int(w * target_longest_edge / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(filepath, "JPEG", quality=95)
        return True
    except Exception:
        return False


def preprocess_species_dir(species_dir: Path, target_size: int = TARGET_SIZE) -> dict:
    """Preprocess all images in a single species directory.

    Steps:
      1. Validate each image
      2. Remove invalid images
      3. Resize to target_size longest edge
      4. Deduplicate via SHA256

    Returns dict with stats.
    """
    stats = {"total": 0, "invalid": 0, "duplicates": 0, "kept": 0}
    images = list(species_dir.glob("*.jpg"))
    stats["total"] = len(images)

    # Step 1-2: validate and remove bad images
    valid_images = []
    for img_path in images:
        if validate_image(img_path):
            valid_images.append(img_path)
        else:
            img_path.unlink()
            stats["invalid"] += 1

    # Step 3: resize
    for img_path in valid_images:
        resize_image(img_path, target_size)

    # Step 4: deduplicate
    seen_hashes = {}
    for img_path in valid_images:
        h = compute_sha256(img_path)
        if h in seen_hashes:
            img_path.unlink()
            stats["duplicates"] += 1
        else:
            seen_hashes[h] = img_path
            stats["kept"] += 1

    return stats


def preprocess_all(raw_dir: str = "data/raw", target_size: int = TARGET_SIZE) -> dict:
    """Preprocess all species directories.

    Returns dict mapping species slug to stats.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    all_stats = {}
    species_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])

    print(f"Preprocessing {len(species_dirs)} species directories...")
    for species_dir in tqdm(species_dirs, desc="Preprocessing"):
        stats = preprocess_species_dir(species_dir, target_size)
        all_stats[species_dir.name] = stats
        if stats["invalid"] > 0 or stats["duplicates"] > 0:
            print(
                f"  {species_dir.name}: {stats['kept']} kept, "
                f"{stats['invalid']} invalid, {stats['duplicates']} duplicates"
            )

    return all_stats

"""Singleton model loader for the Flask web app."""

import os
from pathlib import Path
from typing import Optional

from src.model.inference import TreeClassifier

_classifier: Optional[TreeClassifier] = None


def _download_checkpoint(url: str, dest: str) -> None:
    """Download model checkpoint from a URL (e.g. GitHub Releases)."""
    import requests

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model checkpoint from {url}...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"  {pct}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end="\r")
    print(f"\nCheckpoint saved to {dest}")


def get_classifier() -> TreeClassifier:
    """Get or create the singleton TreeClassifier instance.

    Loads the model from the checkpoint path specified by the
    MODEL_CHECKPOINT environment variable, defaulting to
    'checkpoints/best_model.pt'.

    If the checkpoint file doesn't exist and CHECKPOINT_URL is set,
    downloads it automatically (for cloud deployments like Render).
    """
    global _classifier
    if _classifier is None:
        checkpoint_path = os.environ.get(
            "MODEL_CHECKPOINT", "checkpoints/best_model.pt"
        )

        if not Path(checkpoint_path).exists():
            checkpoint_url = os.environ.get("CHECKPOINT_URL")
            if checkpoint_url:
                _download_checkpoint(checkpoint_url, checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found at {checkpoint_path}. "
                    "Either train a model first (python scripts/train.py) or set "
                    "CHECKPOINT_URL to download one automatically."
                )

        print(f"Loading model from {checkpoint_path}...")
        _classifier = TreeClassifier(checkpoint_path)
        print("Model loaded successfully")
    return _classifier

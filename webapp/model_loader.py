"""Singleton model loader for the Flask web app."""

import os

from src.model.inference import TreeClassifier

_classifier: TreeClassifier | None = None


def get_classifier() -> TreeClassifier:
    """Get or create the singleton TreeClassifier instance.

    Loads the model from the checkpoint path specified by the
    MODEL_CHECKPOINT environment variable, defaulting to
    'checkpoints/best_model.pt'.
    """
    global _classifier
    if _classifier is None:
        checkpoint_path = os.environ.get(
            "MODEL_CHECKPOINT", "checkpoints/best_model.pt"
        )
        print(f"Loading model from {checkpoint_path}...")
        _classifier = TreeClassifier(checkpoint_path)
        print("Model loaded successfully")
    return _classifier

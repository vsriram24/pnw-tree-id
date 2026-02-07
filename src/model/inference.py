"""Single-image inference wrapper for the tree identification model."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset.transforms import get_inference_transforms
from src.model.architecture import create_model


class TreeClassifier:
    """Wrapper for loading a trained model and running predictions."""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to saved .pt checkpoint.
            device: Device string ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        num_classes = checkpoint["num_classes"]

        self.model = create_model(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transforms = get_inference_transforms()

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 5) -> list[dict]:
        """Predict species from an image file.

        Args:
            image_path: Path to the image file.
            top_k: Number of top predictions to return.

        Returns:
            List of dicts with 'species', 'confidence' keys, sorted by confidence.
        """
        img = Image.open(image_path).convert("RGB")
        return self.predict_pil(img, top_k=top_k)

    @torch.no_grad()
    def predict_pil(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """Predict species from a PIL Image.

        Args:
            image: PIL Image (RGB).
            top_k: Number of top predictions to return.

        Returns:
            List of dicts with 'species', 'confidence' keys, sorted by confidence.
        """
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = probs.topk(top_k)
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "species": self.idx_to_class[idx.item()],
                "confidence": round(prob.item(), 4),
            })

        return results

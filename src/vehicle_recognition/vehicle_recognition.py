"""Baseline vehicle recognition using ImageNet-pretrained ResNet18."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


try:  # pragma: no cover - optional dependency
    import torch
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    transforms = None  # type: ignore
    ResNet18_Weights = None  # type: ignore
    resnet18 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


class VehicleRecognizer:
    """Wrapper around ResNet18 to produce top-k vehicle predictions."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.available = torch is not None and resnet18 is not None and Image is not None
        if not self.available:
            self.model = None
            self.transforms = None
            self.categories: List[str] = []
            return

        try:
            weights = ResNet18_Weights.DEFAULT if ResNet18_Weights is not None else None
        except Exception:  # pragma: no cover
            weights = None
        if weights is not None:
            self.model = resnet18(weights=weights).to(device)
            self.transforms = weights.transforms()
            self.categories = list(weights.meta.get("categories", []))
        else:
            self.model = resnet18(weights=None).to(device)
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.categories = [f"class_{i}" for i in range(1000)]
        self.model.eval()

    def predict(self, image_path: Path, topk: int = 3) -> List[Tuple[str, float]]:  # pragma: no cover - heavy dependency
        if not self.available or self.model is None or self.transforms is None:
            raise RuntimeError("Torch/torchvision are required for vehicle recognition")
        image = Image.open(image_path).convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, topk)
        predictions = []
        for score, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            label = self.categories[int(idx)] if self.categories else str(int(idx))
            predictions.append((label, float(score)))
        return predictions


def predict_vehicle(image_path: Path, topk: int = 3, device: str = "cpu") -> List[Tuple[str, float]]:
    recognizer = VehicleRecognizer(device=device)
    return recognizer.predict(image_path, topk=topk)


__all__ = ["VehicleRecognizer", "predict_vehicle"]

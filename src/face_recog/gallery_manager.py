"""Helpers for building a face recognition gallery."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import face_recognition
except Exception:  # pragma: no cover
    face_recognition = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def load_image(path: Path) -> Optional[np.ndarray]:
    if face_recognition is not None:
        return face_recognition.load_image_file(str(path))
    if Image is not None:
        return np.array(Image.open(path).convert("RGB"))
    return None


def build_gallery(
    root: Path = Path("data/known_faces"),
    output: Path = Path("data/gallery_encodings.pkl"),
    model: str = "hog",
) -> Dict[str, np.ndarray]:
    if face_recognition is None:
        raise RuntimeError(
            "The face_recognition package is required. On systems where dlib is "
            "difficult to install consider using alternatives such as insightface "
            "or deepface and adapting this helper accordingly."
        )

    gallery: Dict[str, List[np.ndarray]] = {}
    for person_dir in sorted(root.glob("*/images")):
        person_name = person_dir.parent.name
        encodings: List[np.ndarray] = []
        for image_path in sorted(person_dir.glob("*.jpg")):
            image = load_image(image_path)
            if image is None:
                continue
            try:
                faces = face_recognition.face_encodings(image, model=model)
            except Exception:
                faces = []
            encodings.extend(faces)
        if encodings:
            gallery[person_name] = encodings

    mean_gallery: Dict[str, np.ndarray] = {}
    for name, vectors in gallery.items():
        stacked = np.vstack(vectors)
        mean = stacked.mean(axis=0)
        mean_gallery[name] = mean

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(mean_gallery, f)
    return mean_gallery


__all__ = ["build_gallery"]


if __name__ == "__main__":  # pragma: no cover
    build_gallery()

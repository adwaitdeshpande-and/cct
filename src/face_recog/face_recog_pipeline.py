"""Face recognition pipeline that matches person tracks against a gallery."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import face_recognition
except Exception:  # pragma: no cover
    face_recognition = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


@dataclass
class MatchResult:
    track_id: str
    image_path: str
    match_name: str
    similarity: float
    face_index: int
    bbox: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "image_path": self.image_path,
            "match_name": self.match_name,
            "similarity": self.similarity,
            "face_index": self.face_index,
            "bbox": self.bbox,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_gallery(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        return pickle.load(f)


def run_pipeline(
    gallery_path: Path,
    crops_root: Path,
    outdir: Path,
    threshold: float = 0.6,
    model: str = "hog",
) -> Dict[str, object]:
    if face_recognition is None:
        raise RuntimeError("face_recognition is required for the face pipeline")

    gallery = load_gallery(gallery_path)
    outdir.mkdir(parents=True, exist_ok=True)
    matches_dir = outdir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    results: List[MatchResult] = []

    for track_dir in sorted((crops_root / "person").glob("track_*")):
        track_id = track_dir.name.replace("track_", "")
        track_matches_dir = matches_dir / f"track_{track_id}"
        track_matches_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(track_dir.glob("*.jpg")):
            image = face_recognition.load_image_file(str(image_path))
            locations = face_recognition.face_locations(image, model=model)
            encodings = face_recognition.face_encodings(image, known_face_locations=locations)
            for idx, (bbox, encoding) in enumerate(zip(locations, encodings)):
                best_name = ""
                best_score = -1.0
                for name, gallery_encoding in gallery.items():
                    score = cosine_similarity(encoding, gallery_encoding)
                    if score > best_score:
                        best_score = score
                        best_name = name
                if best_score >= threshold:
                    top, right, bottom, left = bbox
                    face_crop = image[top:bottom, left:right]
                    if Image is not None:
                        Image.fromarray(face_crop).save(track_matches_dir / f"match_{idx}_{image_path.stem}.jpg")
                    result = MatchResult(
                        track_id=track_id,
                        image_path=str(image_path),
                        match_name=best_name,
                        similarity=float(best_score),
                        face_index=idx,
                        bbox=[int(left), int(top), int(right), int(bottom)],
                    )
                    results.append(result)

    results_json = [match.to_dict() for match in results]
    (outdir / "results.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "image_path", "match_name", "similarity", "face_index", "bbox"])
        writer.writeheader()
        for item in results_json:
            writer.writerow(item)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gallery_path": str(gallery_path),
        "num_matches": len(results_json),
        "threshold": threshold,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the face recognition pipeline")
    parser.add_argument("--gallery", default=Path("data/gallery_encodings.pkl"), type=Path)
    parser.add_argument("--crops-root", default=Path("outputs/crops"), type=Path)
    parser.add_argument("--outdir", default=Path("outputs/face_recog"), type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--model", default="hog", help="face_recognition model (hog or cnn)")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(
        gallery_path=args.gallery,
        crops_root=args.crops_root,
        outdir=args.outdir,
        threshold=args.threshold,
        model=args.model,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

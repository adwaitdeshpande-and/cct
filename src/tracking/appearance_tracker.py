"""Appearance assisted tracker using ResNet18 embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torchvision.transforms as T
    from torchvision.models import ResNet18_Weights, resnet18
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    T = None  # type: ignore
    ResNet18_Weights = None  # type: ignore
    resnet18 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


@dataclass
class AppearanceTrack:
    track_id: int
    class_name: str
    last_bbox: List[float]
    last_frame: int
    lost: int = 0
    features: List[np.ndarray] = field(default_factory=list)
    detections: List[Dict[str, object]] = field(default_factory=list)

    def average_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        stacked = np.vstack(self.features)
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "detections": self.detections,
            "num_features": len(self.features),
        }


class FeatureExtractor:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_available = torch is not None and resnet18 is not None
        if not self.model_available:
            self.model = None
            self.transforms = None
            return

        try:
            weights = ResNet18_Weights.DEFAULT if ResNet18_Weights is not None else None
        except Exception:  # pragma: no cover - compatibility with older torchvision
            weights = None

        if weights is not None:
            model = resnet18(weights=weights)
            self.transforms = weights.transforms()
        else:
            model = resnet18(weights=None)
            self.transforms = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)

    def __call__(self, image: Image.Image) -> Optional[np.ndarray]:  # pragma: no cover - heavy dependency
        if not self.model_available or Image is None:
            return None
        assert self.transforms is not None
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        vec = features.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def crop_image(frame_path: Path, bbox: List[float]) -> Optional[Image.Image]:
    if Image is None:
        return None
    with Image.open(frame_path) as img:
        image = img.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(width, int(round(x2)))
    y2 = min(height, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_tracker(
    detections_path: Path,
    frames_dir: Path,
    out_path: Path,
    device: str = "cpu",
    sim_thresh: float = 0.6,
    iou_thresh: float = 0.3,
    max_lost: int = 5,
) -> Dict[str, object]:
    data = json.loads(Path(detections_path).read_text(encoding="utf-8"))
    frames = data.get("frames", [])

    feature_extractor = FeatureExtractor(device=device)

    tracks: List[AppearanceTrack] = []
    finished_tracks: List[AppearanceTrack] = []
    next_track_id = 1

    def retire_lost(current_frame_index: int) -> None:
        nonlocal tracks, finished_tracks
        survivors = []
        for track in tracks:
            if current_frame_index - track.last_frame > max_lost:
                finished_tracks.append(track)
            else:
                survivors.append(track)
        tracks = survivors

    for idx, frame in enumerate(frames):
        frame_index = frame.get("frame_index", idx)
        frame_file = frame.get("frame_file")
        if frame_file is None:
            frame_file = f"frame_{frame_index:06d}.jpg"
        frame_path = frames_dir / frame_file
        detections = frame.get("detections", [])

        for track in tracks:
            track.lost += 1

        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue
            class_name = det.get("class_name", "unknown")
            crop = crop_image(frame_path, bbox) if frame_path.exists() else None
            feature = feature_extractor(crop) if crop is not None else None

            best_track = None
            best_score = -1.0
            for track in tracks:
                if track.class_name != class_name:
                    continue
                if feature is not None:
                    avg = track.average_feature()
                    if avg is not None:
                        score = cosine_similarity(avg, feature)
                        if score > best_score:
                            best_score = score
                            best_track = track
                        continue
                score = iou(track.last_bbox, bbox)
                if score > best_score:
                    best_score = score
                    best_track = track

            matched = False
            if best_track and feature is not None:
                avg = best_track.average_feature()
                if avg is not None and cosine_similarity(avg, feature) >= sim_thresh:
                    matched = True
            if best_track and not matched and feature is None and best_score >= iou_thresh:
                matched = True

            if matched and best_track is not None:
                target = best_track
            else:
                target = AppearanceTrack(
                    track_id=next_track_id,
                    class_name=class_name,
                    last_bbox=list(bbox),
                    last_frame=frame_index,
                )
                next_track_id += 1
                tracks.append(target)

            target.last_bbox = list(bbox)
            target.last_frame = frame_index
            target.lost = 0
            if feature is not None:
                target.features.append(feature)
            target.detections.append(
                {
                    "frame_index": frame_index,
                    "frame_file": frame_file,
                    "timestamp_sec": frame.get("timestamp_sec"),
                    "bbox": bbox,
                    "confidence": det.get("confidence"),
                }
            )

        retire_lost(frame_index)

    finished_tracks.extend(tracks)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_available": feature_extractor.model_available,
        "tracks": [track.to_dict() for track in finished_tracks],
        "source_detections": str(detections_path),
        "frames_directory": str(frames_dir),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Appearance assisted tracker")
    parser.add_argument("--detections", required=True, help="detections.json path")
    parser.add_argument("--frames", required=True, help="Directory with frames")
    parser.add_argument("--out", required=True, help="Output track JSON path")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--sim-thresh", type=float, default=0.6, help="Cosine similarity threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="IoU fallback threshold")
    parser.add_argument("--max-lost", type=int, default=5, help="Maximum number of frames a track can be lost")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_tracker(
        detections_path=Path(args.detections),
        frames_dir=Path(args.frames),
        out_path=Path(args.out),
        device=args.device,
        sim_thresh=max(0.0, min(1.0, args.sim_thresh)),
        iou_thresh=max(0.0, min(1.0, args.iou_thresh)),
        max_lost=max(1, args.max_lost),
    )


if __name__ == "__main__":  # pragma: no cover
    main()

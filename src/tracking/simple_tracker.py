"""Simple IoU based multi-object tracker."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional



@dataclass
class Track:
    track_id: int
    class_name: str
    last_bbox: List[float]
    last_frame: int
    age: int = 0
    detections: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "detections": self.detections,
        }


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


def run_tracker(
    detections_path: Path,
    out_path: Path,
    iou_thresh: float = 0.3,
    max_age: int = 3,
) -> Dict[str, object]:
    data = json.loads(Path(detections_path).read_text(encoding="utf-8"))
    frames = data.get("frames", [])

    next_track_id = 1
    active_tracks: List[Track] = []
    finished_tracks: List[Track] = []

    def flush_old_tracks(current_frame_index: int) -> None:
        nonlocal active_tracks, finished_tracks
        still_active = []
        for track in active_tracks:
            if current_frame_index - track.last_frame > max_age:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

    for frame_idx, frame in enumerate(frames):
        frame_index = frame.get("frame_index", frame_idx)
        frame_file = frame.get("frame_file")
        timestamp = frame.get("timestamp_sec")
        detections = frame.get("detections", [])

        for track in active_tracks:
            track.age += 1

        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue
            class_name = det.get("class_name", "unknown")
            best_iou = 0.0
            best_track: Optional[Track] = None
            for track in active_tracks:
                if track.class_name != class_name:
                    continue
                score = iou(track.last_bbox, bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track and best_iou >= iou_thresh:
                target = best_track
            else:
                target = Track(
                    track_id=next_track_id,
                    class_name=class_name,
                    last_bbox=list(bbox),
                    last_frame=frame_index,
                )
                next_track_id += 1
                active_tracks.append(target)

            target.last_bbox = list(bbox)
            target.last_frame = frame_index
            target.age = 0
            record = {
                "frame_index": frame_index,
                "frame_file": frame_file,
                "timestamp_sec": timestamp,
                "bbox": bbox,
                "confidence": det.get("confidence"),
            }
            target.detections.append(record)

        flush_old_tracks(frame_index)

    finished_tracks.extend(active_tracks)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_detections": str(detections_path),
        "tracks": [track.to_dict() for track in finished_tracks],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple IoU tracker")
    parser.add_argument("--detections", required=True, help="detections.json file")
    parser.add_argument("--out", required=True, help="tracks.json output path")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--max-age", type=int, default=3, help="Maximum allowed age for tracks")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_tracker(
        detections_path=Path(args.detections),
        out_path=Path(args.out),
        iou_thresh=max(0.0, min(1.0, args.iou)),
        max_age=max(1, args.max_age),
    )


if __name__ == "__main__":  # pragma: no cover
    main()

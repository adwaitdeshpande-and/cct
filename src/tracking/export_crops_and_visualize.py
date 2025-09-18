"""Export cropped detections and visualise tracks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def deterministic_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=track_id)
    r, g, b = rng.integers(0, 255, size=3)
    return int(r), int(g), int(b)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_and_save(image: np.ndarray, bbox: List[float], out_path: Path) -> None:
    x1, y1, x2, y2 = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    crop = image[y1:y2, x1:x2]
    ensure_dir(out_path.parent)
    if cv2 is not None:
        cv2.imwrite(str(out_path), crop)
    else:  # pragma: no cover
        Image.fromarray(crop[:, :, ::-1]).save(out_path)


def annotate_frame(frame: np.ndarray, annotations: List[Dict[str, object]]) -> np.ndarray:
    annotated = frame.copy()
    for ann in annotations:
        bbox = ann["bbox"]
        label = ann["label"]
        color = ann["color"]
        x1, y1, x2, y2 = map(int, bbox)
        if cv2 is not None:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        else:  # pragma: no cover
            pil_img = Image.fromarray(annotated)
            pil_img = pil_img.convert("RGB")
            annotated = np.array(pil_img)
    return annotated


def run_export(
    tracks_path: Path,
    frames_dir: Path,
    outdir: Path,
    video_filename: str = "annotated_tracks.mp4",
    fps: float = 25.0,
) -> Dict[str, object]:
    data = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])

    ensure_dir(outdir)
    crops_dir = outdir / "crops"
    annotated_frames_dir = outdir / "annotated_frames"
    ensure_dir(crops_dir)
    ensure_dir(annotated_frames_dir)

    frame_annotations: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for track in tracks:
        track_id = int(track.get("track_id", 0))
        class_name = track.get("class_name", "object")
        color = deterministic_color(track_id)
        for det in track.get("detections", []):
            frame_index = int(det.get("frame_index", 0))
            frame_file = det.get("frame_file") or f"frame_{frame_index:06d}.jpg"
            bbox = det.get("bbox")
            if bbox is None:
                continue
            frame_annotations[frame_index].append(
                {
                    "frame_file": frame_file,
                    "bbox": bbox,
                    "label": f"ID{track_id}:{class_name}",
                    "color": color,
                    "class_name": class_name,
                    "track_id": track_id,
                }
            )

    written_frames = []

    for frame_index in sorted(frame_annotations.keys()):
        annotations = frame_annotations[frame_index]
        frame_file = annotations[0]["frame_file"]
        frame_path = frames_dir / frame_file
        if cv2 is not None and frame_path.exists():
            frame = cv2.imread(str(frame_path))
        elif frame_path.exists() and Image is not None:  # pragma: no cover
            frame = np.array(Image.open(frame_path).convert("RGB"))[:, :, ::-1]
        else:
            continue

        for ann in annotations:
            track_id = ann["track_id"]
            class_name = ann["class_name"]
            bbox = ann["bbox"]
            crop_path = crops_dir / class_name / f"track_{track_id:04d}" / f"frame_{frame_index:06d}.jpg"
            crop_and_save(frame, bbox, crop_path)

        annotated_frame = annotate_frame(frame, annotations)
        out_frame_path = annotated_frames_dir / f"frame_{frame_index:06d}.jpg"
        if cv2 is not None:
            cv2.imwrite(str(out_frame_path), annotated_frame)
        else:  # pragma: no cover
            Image.fromarray(annotated_frame[:, :, ::-1]).save(out_frame_path)
        written_frames.append((frame_index, annotated_frame))

    video_path = outdir / video_filename
    if cv2 is not None and written_frames:
        height, width = written_frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        for _, frame in written_frames:
            writer.write(frame)
        writer.release()

    summary = {
        "tracks_file": str(tracks_path),
        "frames_dir": str(frames_dir),
        "crops_dir": str(crops_dir),
        "annotated_frames_dir": str(annotated_frames_dir),
        "video_path": str(video_path) if video_path.exists() else None,
        "num_frames": len(written_frames),
    }
    (outdir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export crops and annotated video from tracks")
    parser.add_argument("--tracks", required=True, help="tracks.json path")
    parser.add_argument("--frames", required=True, help="Directory with extracted frames")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--video-filename", default="annotated_tracks.mp4", help="Annotated video filename")
    parser.add_argument("--fps", type=float, default=25.0, help="Frame rate for annotated video")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_export(
        tracks_path=Path(args.tracks),
        frames_dir=Path(args.frames),
        outdir=Path(args.outdir),
        video_filename=args.video_filename,
        fps=args.fps,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

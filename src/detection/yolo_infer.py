"""YOLO based object detection helper.

The script can operate on a directory of frames or on a single video
file.  The heavy lifting is delegated to :mod:`ultralytics`; however the
implementation keeps generous error handling and fallbacks so that the
rest of the pipeline can continue even when the dependency is missing on
the current platform.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


@dataclass
class Detection:
    bbox: List[float]
    confidence: float
    class_id: int

    @property
    def class_name(self) -> str:
        return COCO_NAMES.get(self.class_id, f"class_{self.class_id}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "bbox": self.bbox,
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


def _load_model(model_path: str) -> Optional[YOLO]:
    if YOLO is None:  # pragma: no cover - requires heavy dependency
        return None
    model = YOLO(model_path)
    return model


def _annotate(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    if cv2 is None:
        return image
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name}:{det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _gather_frames(frames_path: Optional[Path]) -> List[Path]:
    if not frames_path:
        return []
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames path does not exist: {frames_path}")
    files = [p for p in sorted(frames_path.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return files


def _video_iterator(video_path: Path):
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for processing video files")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_index / fps if fps > 0 else float(frame_index)
        yield frame_index, frame, timestamp
        frame_index += 1
    cap.release()


def run_detection(
    frames: List[Path],
    video: Optional[Path],
    outdir: Path,
    model_name: str,
    conf: float = 0.25,
    save_video: bool = False,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    annotated_dir = outdir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    detections_path = outdir / "detections.json"
    frames_payload: List[Dict[str, object]] = []

    model = _load_model(model_name)
    model_available = model is not None

    if video:
        width = height = fps = None
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
        writer = None
        if save_video and cv2 is not None and width and height and fps:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(outdir / "annotated_video.mp4"), fourcc, fps or 25.0, (width, height))

        iterator = _video_iterator(video)
        for frame_index, frame, timestamp in tqdm(iterator, desc="YOLO inference (video)", unit="frame"):
            dets = _predict_frame(model, frame, conf) if model_available else []
            annotated = _annotate(frame, dets)
            if writer is not None:
                writer.write(annotated)

            if annotated_dir:
                img_path = annotated_dir / f"frame_{frame_index:06d}.jpg"
                if cv2 is not None:
                    cv2.imwrite(str(img_path), annotated)
                else:  # pragma: no cover - OpenCV should exist for video path
                    from PIL import Image

                    Image.fromarray(annotated).save(img_path)

            frames_payload.append(
                {
                    "frame_index": frame_index,
                    "frame_file": f"frame_{frame_index:06d}.jpg",
                    "timestamp_sec": float(timestamp),
                    "detections": [det.to_dict() for det in dets],
                }
            )

        if writer is not None:
            writer.release()

    else:
        for idx, frame_path in enumerate(tqdm(frames, desc="YOLO inference (frames)", unit="frame")):
            image = _read_image(frame_path)
            dets = _predict_frame(model, image, conf) if model_available else []
            annotated = _annotate(image, dets)
            out_path = annotated_dir / frame_path.name
            if cv2 is not None:
                cv2.imwrite(str(out_path), annotated)
            else:  # pragma: no cover - OpenCV normally installed
                from PIL import Image

                Image.fromarray(annotated).save(out_path)

            frames_payload.append(
                {
                    "frame_index": idx,
                    "frame_file": frame_path.name,
                    "timestamp_sec": float(idx),
                    "detections": [det.to_dict() for det in dets],
                }
            )

    payload = {
        "model": model_name,
        "model_available": model_available,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "frames": frames_payload,
    }

    detections_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _read_image(path: Path) -> np.ndarray:
    if cv2 is None:  # pragma: no cover
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))
    return cv2.imread(str(path))


def _predict_frame(model, frame: np.ndarray, conf: float) -> List[Detection]:  # pragma: no cover - requires YOLO
    if model is None:
        return []

    results = model.predict(source=frame, conf=conf, verbose=False)
    detections: List[Detection] = []
    for result in results:
        if not hasattr(result, "boxes"):
            continue
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cls = int(box.cls[0]) if hasattr(box, "cls") else 0
            conf_val = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            detections.append(Detection(bbox=xyxy, confidence=conf_val, class_id=cls))
    return detections


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO object detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frames", help="Directory containing frames")
    group.add_argument("--video", help="Video file to process")
    parser.add_argument("--model", default="yolov8n.pt", help="Model name or path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video when processing --video")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    frames_dir = Path(args.frames) if args.frames else None
    video_path = Path(args.video) if args.video else None

    if frames_dir:
        frames = _gather_frames(frames_dir)
        video = None
    else:
        frames = []
        video = video_path

    run_detection(
        frames=frames,
        video=video,
        outdir=Path(args.outdir),
        model_name=args.model,
        conf=max(0.0, min(1.0, args.conf)),
        save_video=args.save_video,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

"""Video preprocessing utilities.

This module extracts frames from an input video, records hashing
information and stores metadata artefacts required for later stages of
the pipeline.  The implementation intentionally favours readability so
that investigators can audit each step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

try:  # pragma: no cover - optional dependency; behaviour tested via integration test
    import cv2
except Exception:  # pragma: no cover - covered indirectly through tests when cv2 available
    cv2 = None  # type: ignore

from ..utils.hashing import sha256_bytes, sha256_file
from .buftool_wrapper import run_buftool


@dataclass
class FrameInfo:
    """Container describing metadata for a single extracted frame."""

    frame_index: int
    timestamp_sec: float
    filename: str
    sha256: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "filename": self.filename,
            "sha256": self.sha256,
        }


def _ensure_opencv_available() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for preprocessing but is not installed. "
            "Install opencv-python or opencv-python-headless."
        )


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _extract_frames(
    video_path: Path,
    frames_dir: Path,
    frame_skip: int = 0,
    max_frames: Optional[int] = None,
) -> tuple[List[FrameInfo], float, int]:
    """Extract frames from *video_path* into *frames_dir*.

    ``frame_skip`` controls how many frames are skipped between saved
    frames (0 means save every frame).  ``max_frames`` limits the number
    of frames that will be written to disk.
    """

    _ensure_opencv_available()
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    manifest_frames: List[FrameInfo] = []

    frame_index = 0
    saved = 0

    with tqdm(total=total_frames or None, desc="Extracting frames", unit="frame") as pbar:
        while True:
            if max_frames is not None and saved >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                frame_index += 1
                pbar.update(1)
                continue

            timestamp = frame_index / fps if fps > 0 else float(frame_index)
            file_name = f"frame_{frame_index:06d}.jpg"
            save_path = frames_dir / file_name

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError(f"Failed to encode frame {frame_index}")
            data = buffer.tobytes()
            sha = sha256_bytes(data)
            save_path.write_bytes(data)

            manifest_frames.append(
                FrameInfo(
                    frame_index=frame_index,
                    timestamp_sec=float(timestamp),
                    filename=file_name,
                    sha256=sha,
                )
            )
            saved += 1
            frame_index += 1
            pbar.update(1)

    cap.release()

    return manifest_frames, fps, total_frames


def preprocess(
    input_path: Path,
    outdir: Path,
    frame_skip: int = 0,
    max_frames: Optional[int] = None,
    use_exiftool: bool = False,
    use_buftool: bool = False,
) -> Dict[str, object]:
    """Main preprocessing entry point used by the CLI."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / "frames"

    original_sha = sha256_file(input_path)
    _write_text(outdir / "original_sha256.txt", original_sha)

    frames, fps, total_frames = _extract_frames(input_path, frames_dir, frame_skip, max_frames)

    frame_hashes = {frame.filename: frame.sha256 for frame in frames}
    _save_json(outdir / "frame_hashes.json", frame_hashes)

    manifest: Dict[str, object] = {
        "input": str(input_path.resolve()),
        "output_directory": str(outdir.resolve()),
        "sha256": original_sha,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "fps": fps,
        "total_frames": total_frames,
        "saved_frames": len(frames),
        "frame_skip": frame_skip,
        "max_frames": max_frames,
        "use_exiftool": use_exiftool,
        "use_buftool": use_buftool,
        "frames": [frame.to_dict() for frame in frames],
    }

    exiftool_path = outdir / "exiftool.txt"
    if use_exiftool:
        executable = shutil.which("exiftool")
        if executable:
            try:
                result = subprocess.run(
                    [executable, str(input_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                exiftool_path.write_text(result.stdout, encoding="utf-8")
                manifest["exiftool_output"] = str(exiftool_path)
                if result.returncode != 0:
                    manifest["exiftool_warning"] = result.stderr.strip()
            except OSError as exc:  # pragma: no cover
                manifest["exiftool_error"] = str(exc)
        else:
            manifest["exiftool_error"] = "exiftool_not_found"
    else:
        manifest["exiftool_output"] = None

    if use_buftool:
        buftool_result = run_buftool(input_path)
        buftool_path = outdir / "buftool.json"
        if buftool_result:
            _save_json(buftool_path, buftool_result)
            manifest["buftool_output"] = str(buftool_path)
    else:
        manifest["buftool_output"] = None

    manifest_path = outdir / "manifest.json"
    _save_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess multimedia evidence")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--frame-skip", type=int, default=0, help="Number of frames to skip between saves")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to extract")
    parser.add_argument("--use-exiftool", action="store_true", help="Run exiftool and store output")
    parser.add_argument("--use-buftool", action="store_true", help="Run buftool and store output")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    preprocess(
        input_path=input_path,
        outdir=outdir,
        frame_skip=max(args.frame_skip, 0),
        max_frames=args.max_frames,
        use_exiftool=args.use_exiftool,
        use_buftool=args.use_buftool,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

"""Basic tamper detection utilities."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from ..utils.hashing import sha256_file


def compute_prnu_residual(image: np.ndarray) -> np.ndarray:
    """Compute a simple noise residual used as a PRNU placeholder."""

    if cv2 is None:
        return image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    residual = image.astype(np.float32) - blurred.astype(np.float32)
    return residual


def normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    if a_flat.size == 0 or b_flat.size == 0:
        return 0.0
    a_norm = (a_flat - a_flat.mean())
    b_norm = (b_flat - b_flat.mean())
    denom = np.linalg.norm(a_norm) * np.linalg.norm(b_norm)
    if denom == 0:
        return 0.0
    return float(np.dot(a_norm, b_norm) / denom)


def load_image(path: Path) -> Optional[np.ndarray]:
    if cv2 is not None:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    try:
        from PIL import Image
    except Exception:  # pragma: no cover
        return None
    return np.array(Image.open(path).convert("L"))


def parse_exiftool_output(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def check_sha256(manifest: Dict[str, object], input_path: Path, report: Dict[str, object]) -> None:
    expected = manifest.get("sha256")
    if expected is None:
        report.setdefault("flags", []).append("manifest_missing_sha256")
        return
    actual = sha256_file(input_path)
    if actual != expected:
        report.setdefault("flags", []).append("sha256_mismatch")
    report.setdefault("checks", {})["sha256"] = {"expected": expected, "actual": actual}


def check_metadata(manifest: Dict[str, object], input_path: Path, report: Dict[str, object]) -> None:
    exiftool_path = manifest.get("exiftool_output")
    if not exiftool_path:
        return
    data = parse_exiftool_output(Path(exiftool_path))
    file_mtime = datetime.fromtimestamp(input_path.stat().st_mtime, tz=timezone.utc)
    exif_date_str = data.get("Create Date") or data.get("Modify Date")
    if exif_date_str:
        try:
            exif_date = datetime.strptime(exif_date_str, "%Y:%m:%d %H:%M:%S%z")
        except ValueError:
            try:
                exif_date = datetime.strptime(exif_date_str, "%Y:%m:%d %H:%M:%S")
                exif_date = exif_date.replace(tzinfo=timezone.utc)
            except ValueError:
                exif_date = None
        if exif_date is not None:
            delta = abs((file_mtime - exif_date).total_seconds())
            report.setdefault("checks", {})["metadata_delta_seconds"] = delta
            if delta > 300:
                report.setdefault("flags", []).append("metadata_time_mismatch")


def check_prnu(frames_dir: Optional[Path], fingerprint_path: Optional[Path], report: Dict[str, object]) -> None:
    if frames_dir is None or fingerprint_path is None:
        return
    if not fingerprint_path.exists():
        report.setdefault("flags", []).append("fingerprint_missing")
        return
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        return
    image = load_image(frame_files[0])
    if image is None:
        return
    residual = compute_prnu_residual(image)
    fingerprint = np.load(fingerprint_path)
    residual_resized = residual
    if fingerprint.shape != residual.shape:
        residual_resized = cv2.resize(residual, (fingerprint.shape[1], fingerprint.shape[0])) if cv2 is not None else residual
    score = normalised_correlation(residual_resized, fingerprint)
    report.setdefault("checks", {})["prnu_score"] = score
    if score < 0.1:
        report.setdefault("flags", []).append("prnu_low_similarity")


def deepfake_stub(input_path: Path) -> float:
    """Placeholder that returns a heuristic deepfake score.

    This implementation simply analyses the average luminance variance as a
    very coarse proxy.  In production an actual detector should be plugged
    in here.
    """

    image = load_image(input_path)
    if image is None:
        return 0.0
    return float(image.var() / (255.0 ** 2))


def run_checks(
    manifest_path: Path,
    input_path: Path,
    frames_dir: Optional[Path] = None,
    fingerprint_path: Optional[Path] = None,
    outdir: Optional[Path] = None,
) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report: Dict[str, object] = {"flags": [], "checks": {}}

    check_sha256(manifest, input_path, report)
    check_metadata(manifest, input_path, report)
    check_prnu(frames_dir, fingerprint_path, report)

    deepfake_score = deepfake_stub(input_path)
    report["checks"]["deepfake_score"] = deepfake_score

    report["tamper_score"] = min(1.0, 0.2 * len(report["flags"]) + deepfake_score)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "tamper_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Perform basic tamper checks")
    parser.add_argument("--manifest", required=True, help="Path to preprocess manifest.json")
    parser.add_argument("--input", required=True, help="Original input file path")
    parser.add_argument("--frames", help="Directory containing extracted frames")
    parser.add_argument("--fingerprint", help="Path to stored PRNU fingerprint (NumPy .npy file)")
    parser.add_argument("--outdir", default="outputs/tamper", help="Directory to write report")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_checks(
        manifest_path=Path(args.manifest),
        input_path=Path(args.input),
        frames_dir=Path(args.frames) if args.frames else None,
        fingerprint_path=Path(args.fingerprint) if args.fingerprint else None,
        outdir=Path(args.outdir) if args.outdir else None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

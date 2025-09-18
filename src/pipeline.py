"""High level orchestrator for the forensic toolkit."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.anpr.anpr_pipeline import process_crops as run_anpr
from src.anpr.aggregate_anpr_results import process as aggregate_anpr
from src.detection.yolo_infer import run_detection
from src.forensics.preprocess import preprocess
from src.packager.manifest import generate_manifest
from src.packager.pdf_report import build_report
from src.tamper.tamper_detector import run_checks as run_tamper
from src.tracking.export_crops_and_visualize import run_export as export_tracks
from src.tracking.simple_tracker import run_tracker as simple_track

try:  # optional
    from src.tracking.appearance_tracker import run_tracker as appearance_track
except Exception:  # pragma: no cover
    appearance_track = None  # type: ignore

try:  # optional
    from src.face_recog.face_recog_pipeline import run_pipeline as run_face_pipeline
except Exception:  # pragma: no cover
    run_face_pipeline = None  # type: ignore


LOGGER = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    preprocess_dir = outdir / "preprocess"
    detection_dir = outdir / "detection"
    tracking_dir = outdir / "tracking"
    crops_dir = outdir / "crops"
    anpr_dir = outdir / "anpr"
    face_dir = outdir / "face_recog"
    tamper_dir = outdir / "tamper"

    LOGGER.info("Starting preprocessing step")
    manifest = preprocess(
        input_path=input_path,
        outdir=preprocess_dir,
        frame_skip=max(args.frame_skip, 0),
        max_frames=args.max_frames,
        use_exiftool=args.use_exiftool,
        use_buftool=args.use_buftool,
    )

    frames_dir = preprocess_dir / "frames"
    modules_run: List[str] = ["preprocess"]

    detection_results = None
    if args.run_detection:
        LOGGER.info("Running detection using model %s", args.detection_model)
        detection_results = run_detection(
            frames=sorted(frames_dir.glob("*.jpg")),
            video=None,
            outdir=detection_dir,
            model_name=args.detection_model,
            conf=args.detection_conf,
            save_video=False,
        )
        modules_run.append("detection")

    tracks_path = None
    if args.run_tracking and detection_results:
        LOGGER.info("Running simple tracker")
        tracks_path = tracking_dir / "tracks.json"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        simple_track(
            detections_path=detection_dir / "detections.json",
            out_path=tracks_path,
            iou_thresh=args.tracker_iou,
            max_age=args.tracker_max_age,
        )
        modules_run.append("tracking")

        LOGGER.info("Exporting crops and annotated frames")
        export_tracks(
            tracks_path=tracks_path,
            frames_dir=frames_dir,
            outdir=outdir,
        )

        if args.run_appearance_tracker and appearance_track is not None:
            LOGGER.info("Running appearance tracker")
            appearance_track(
                detections_path=detection_dir / "detections.json",
                frames_dir=frames_dir,
                out_path=tracking_dir / "tracks_appearance.json",
                device=args.device,
            )

    if args.run_anpr:
        LOGGER.info("Running ANPR module")
        anpr_result = run_anpr(
            crops_root=crops_dir,
            outdir=anpr_dir,
            vehicle_classes=args.vehicle_classes or None,
            preprocess_enabled=not args.anpr_no_preprocess,
            use_tesseract=args.anpr_use_tesseract,
        )
        modules_run.append("anpr")
        if (anpr_dir / "results.json").exists():
            LOGGER.info("Aggregating ANPR results")
            aggregate_anpr(results_path=anpr_dir / "results.json", outdir=anpr_dir)

    if args.run_face and run_face_pipeline is not None:
        LOGGER.info("Running face recognition pipeline")
        run_face_pipeline(
            gallery_path=Path(args.gallery_path),
            crops_root=crops_dir,
            outdir=face_dir,
            threshold=args.face_threshold,
            model=args.face_model,
        )
        modules_run.append("face_recognition")

    if args.run_tamper:
        LOGGER.info("Running tamper checks")
        run_tamper(
            manifest_path=preprocess_dir / "manifest.json",
            input_path=input_path,
            frames_dir=frames_dir,
            fingerprint_path=Path(args.prnu_fingerprint) if args.prnu_fingerprint else None,
            outdir=tamper_dir,
        )
        modules_run.append("tamper")

    if args.package:
        LOGGER.info("Packaging outputs")
        package_manifest = generate_manifest(
            input_path=input_path,
            output_root=outdir,
            modules_run=modules_run,
            parameters={
                "frame_skip": args.frame_skip,
                "max_frames": args.max_frames,
                "detection_model": args.detection_model,
            },
        )
        report_path = outdir / "report.pdf"
        build_report(
            manifest_path=outdir / "package_manifest.json",
            output_path=report_path,
            case_id=args.case_id,
            detections_path=detection_dir / "detections.json",
            tracks_path=tracking_dir / "tracks.json" if tracks_path else None,
            anpr_path=anpr_dir / "aggregated.json" if (anpr_dir / "aggregated.json").exists() else None,
            face_path=face_dir / "results.json" if (face_dir / "results.json").exists() else None,
            tamper_path=tamper_dir / "tamper_report.json" if (tamper_dir / "tamper_report.json").exists() else None,
        )
        modules_run.append("package")

    LOGGER.info("Pipeline completed")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the comprehensive pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--frame-skip", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--use-exiftool", action="store_true")
    parser.add_argument("--use-buftool", action="store_true")
    parser.add_argument("--run-detection", action="store_true")
    parser.add_argument("--run-tracking", action="store_true")
    parser.add_argument("--run-appearance-tracker", action="store_true")
    parser.add_argument("--run-anpr", action="store_true")
    parser.add_argument("--run-face", action="store_true")
    parser.add_argument("--run-tamper", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--detection-model", default="yolov8n.pt")
    parser.add_argument("--detection-conf", type=float, default=0.25)
    parser.add_argument("--tracker-iou", type=float, default=0.3)
    parser.add_argument("--tracker-max-age", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vehicle-classes", nargs="*")
    parser.add_argument("--anpr-no-preprocess", action="store_true")
    parser.add_argument("--anpr-use-tesseract", action="store_true")
    parser.add_argument("--gallery-path", default="data/gallery_encodings.pkl")
    parser.add_argument("--face-threshold", type=float, default=0.6)
    parser.add_argument("--face-model", default="hog")
    parser.add_argument("--prnu-fingerprint")
    parser.add_argument("--case-id", default="CASE-0001")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover
    main()

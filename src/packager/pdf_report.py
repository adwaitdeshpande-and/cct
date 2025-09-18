"""Generate a courtroom friendly PDF report summarising outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def wrap_text(text: str, width: int = 90) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def draw_section(c: canvas.Canvas, title: str, lines: List[str], x: float, y: float) -> float:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, title)
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)
    for line in lines:
        c.drawString(x, y, line)
        y -= 0.4 * cm
        if y < 3 * cm:
            c.showPage()
            y = A4[1] - 2 * cm
            c.setFont("Helvetica", 10)
    return y


def summarise_manifest(manifest: Dict[str, object]) -> List[str]:
    lines = [f"Input file: {manifest.get('input_file')}"]
    lines.append(f"Output root: {manifest.get('output_root')}")
    lines.append(f"Generated at: {manifest.get('generated_at')}")
    lines.append(f"Modules: {', '.join(manifest.get('modules', []))}")
    lines.append(f"Git commit: {manifest.get('git_commit')}")
    return lines


def summarise_detections(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return ["Detections file not provided."]
    data = json.loads(path.read_text(encoding="utf-8"))
    total = sum(len(frame.get("detections", [])) for frame in data.get("frames", []))
    lines = [f"Model: {data.get('model')} (available={data.get('model_available')})"]
    lines.append(f"Frames processed: {len(data.get('frames', []))}")
    lines.append(f"Total detections: {total}")
    return lines


def summarise_tracks(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return ["Tracks file not provided."]
    data = json.loads(path.read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])
    lines = [f"Tracks generated: {len(tracks)}"]
    top = min(5, len(tracks))
    for track in tracks[:top]:
        lines.append(
            f"Track {track.get('track_id')}: {track.get('class_name')} with {len(track.get('detections', []))} observations"
        )
    return lines


def summarise_anpr(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return ["ANPR results unavailable."]
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"Total plate hypotheses: {len(data)}"]
    top = min(5, len(data))
    for entry in data[:top]:
        lines.append(
            f"Track {entry.get('track_id')} ({entry.get('vehicle_class')}): {entry.get('text')} ({entry.get('confidence'):.2f})"
        )
    return lines


def summarise_faces(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return ["Face recognition results unavailable."]
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"Total face matches: {len(data)}"]
    for entry in data[:5]:
        lines.append(
            f"Track {entry.get('track_id')} matched {entry.get('match_name')} with similarity {entry.get('similarity'):.2f}"
        )
    return lines


def summarise_tamper(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return ["Tamper report unavailable."]
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"Tamper score: {data.get('tamper_score')}"]
    if data.get("flags"):
        lines.append("Flags: " + ", ".join(data.get("flags", [])))
    return lines


def build_report(
    manifest_path: Path,
    output_path: Path,
    case_id: str,
    detections_path: Optional[Path] = None,
    tracks_path: Optional[Path] = None,
    anpr_path: Optional[Path] = None,
    face_path: Optional[Path] = None,
    tamper_path: Optional[Path] = None,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4

    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "Case Report")
    y -= 0.7 * cm
    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, y, f"Case ID: {case_id}")
    y -= 1 * cm

    y = draw_section(c, "Manifest", summarise_manifest(manifest), 2 * cm, y)
    y = draw_section(c, "Detections", summarise_detections(detections_path), 2 * cm, y)
    y = draw_section(c, "Tracking", summarise_tracks(tracks_path), 2 * cm, y)
    y = draw_section(c, "ANPR", summarise_anpr(anpr_path), 2 * cm, y)
    y = draw_section(c, "Face Recognition", summarise_faces(face_path), 2 * cm, y)
    y = draw_section(c, "Tamper Analysis", summarise_tamper(tamper_path), 2 * cm, y)

    custody = manifest.get("chain_of_custody", {})
    custody_lines = ["Chain of custody template:"]
    for key in ["processed_by", "reviewed_by", "processing_signature", "notes"]:
        custody_lines.append(f"{key}: {custody.get(key, '')}")
    draw_section(c, "Chain of Custody", custody_lines, 2 * cm, y)

    c.showPage()
    c.save()
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PDF report")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--detections")
    parser.add_argument("--tracks")
    parser.add_argument("--anpr")
    parser.add_argument("--faces")
    parser.add_argument("--tamper")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    build_report(
        manifest_path=Path(args.manifest),
        output_path=Path(args.out),
        case_id=args.case_id,
        detections_path=Path(args.detections) if args.detections else None,
        tracks_path=Path(args.tracks) if args.tracks else None,
        anpr_path=Path(args.anpr) if args.anpr else None,
        face_path=Path(args.faces) if args.faces else None,
        tamper_path=Path(args.tamper) if args.tamper else None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

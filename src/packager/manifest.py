"""Build a consolidated manifest for pipeline outputs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..utils.hashing import sha256_file


OPTIONAL_LIBS = ["torch", "torchvision", "ultralytics", "opencv", "face_recognition", "easyocr"]


def detect_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in OPTIONAL_LIBS:
        try:  # pragma: no cover - optional
            module = __import__(name)
            version = getattr(module, "__version__", "unknown")
        except Exception:
            version = "not_available"
        versions[name] = version
    return versions


def list_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(path)
    return files


def generate_manifest(
    input_path: Path,
    output_root: Path,
    modules_run: List[str],
    parameters: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    files_info = []
    for file_path in list_files(output_root):
        rel = file_path.relative_to(output_root)
        files_info.append({"path": str(rel), "sha256": sha256_file(file_path)})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path.resolve()),
        "output_root": str(output_root.resolve()),
        "modules": modules_run,
        "parameters": parameters or {},
        "files": files_info,
        "tool_versions": detect_versions(),
        "git_commit": os.environ.get("GIT_COMMIT", "GIT_COMMIT_PLACEHOLDER"),
        "chain_of_custody": {
            "processed_by": "",
            "reviewed_by": "",
            "processing_signature": "",
            "notes": "",
        },
    }
    manifest_path = output_root / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate consolidated manifest")
    parser.add_argument("--input", required=True, help="Original input file path")
    parser.add_argument("--outdir", required=True, help="Output directory root")
    parser.add_argument("--modules", nargs="*", default=[], help="List of modules executed")
    parser.add_argument("--params", help="Optional JSON string with parameters")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    parameters = json.loads(args.params) if args.params else {}
    generate_manifest(
        input_path=Path(args.input),
        output_root=Path(args.outdir),
        modules_run=args.modules,
        parameters=parameters,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

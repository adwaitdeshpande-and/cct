"""Automatic number plate recognition utilities."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependencies
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None  # type: ignore

try:  # pragma: no cover
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


PLATE_REGEX = re.compile(r"[A-Z0-9]{3,}")


def locate_plate_candidates(image: np.ndarray) -> List[List[int]]:
    if cv2 is None:
        h, w = image.shape[:2]
        return [[0, 0, w, h]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[List[int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h or 1)
        area = w * h
        if area < 200 or aspect < 2 or aspect > 7:
            continue
        candidates.append([x, y, x + w, y + h])
    candidates.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    return candidates[:5] or [[0, 0, image.shape[1], image.shape[0]]]


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return image
    upscaled = cv2.resize(image, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(filtered, -1, kernel)
    return sharpened


def run_easyocr(reader, image: np.ndarray) -> List[Dict[str, object]]:  # pragma: no cover - heavy dependency
    results = []
    if reader is None:
        return results
    ocr_results = reader.readtext(image)
    for bbox, text, conf in ocr_results:
        cleaned = normalise_plate(text)
        if cleaned:
            results.append({"text": cleaned, "confidence": float(conf), "method": "easyocr"})
    return results


def run_tesseract(image: np.ndarray) -> List[Dict[str, object]]:
    results = []
    if pytesseract is None or Image is None:
        return results
    if image.ndim == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cv2 is not None else image[:, :, ::-1]
    else:
        rgb = image
    pil_img = Image.fromarray(rgb)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = 0.0
        cleaned = normalise_plate(text)
        if cleaned:
            results.append({"text": cleaned, "confidence": conf_val / 100.0, "method": "tesseract"})
    return results


def normalise_plate(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    match = PLATE_REGEX.search(cleaned)
    return match.group(0) if match else ""


def run_ocr(image: np.ndarray, use_tesseract: bool, easyocr_reader) -> List[Dict[str, object]]:
    if not use_tesseract and easyocr_reader is not None:
        results = run_easyocr(easyocr_reader, image)
        if results:
            return results
    return run_tesseract(image)


def process_crops(
    crops_root: Path,
    outdir: Path,
    vehicle_classes: Optional[List[str]] = None,
    preprocess_enabled: bool = True,
    use_tesseract: bool = False,
) -> Dict[str, object]:
    ensure_dirs(outdir)
    plates_dir = outdir / "plates"
    ensure_dirs(plates_dir)

    reader = None
    if not use_tesseract and easyocr is not None:
        try:  # pragma: no cover - easyocr heavy
            reader = easyocr.Reader(["en"], gpu=False)
        except Exception:
            reader = None

    results: List[Dict[str, object]] = []

    classes_to_scan = vehicle_classes or [p.name for p in crops_root.iterdir() if p.is_dir()]

    for vehicle_class in classes_to_scan:
        class_dir = crops_root / vehicle_class
        if not class_dir.exists():
            continue
        for track_dir in sorted(class_dir.glob("track_*")):
            track_id = track_dir.name.replace("track_", "")
            plate_output_dir = plates_dir / f"track_{track_id}"
            ensure_dirs(plate_output_dir)
            for crop_path in sorted(track_dir.glob("*.jpg")):
                image = load_image(crop_path)
                if image is None:
                    continue
                candidates = locate_plate_candidates(image)
                for idx, box in enumerate(candidates):
                    x1, y1, x2, y2 = box
                    plate_img = image[y1:y2, x1:x2]
                    processed = preprocess_plate(plate_img) if preprocess_enabled else plate_img
                    if processed.ndim == 2:
                        save_img = processed
                        if cv2 is not None:
                            ocr_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                        else:
                            ocr_image = np.stack([processed] * 3, axis=-1)
                    else:
                        save_img = processed if cv2 is not None else processed[:, :, ::-1]
                        ocr_image = processed
                    plate_file = plate_output_dir / f"{crop_path.stem}_cand{idx}.png"
                    save_plate_image(save_img, plate_file)
                    ocr_results = run_ocr(ocr_image, use_tesseract, reader)
                    if not ocr_results:
                        results.append(
                            {
                                "track_id": track_id,
                                "vehicle_class": vehicle_class,
                                "crop_image": str(crop_path),
                                "plate_image": str(plate_file),
                                "text": "",
                                "confidence": 0.0,
                                "method": "none",
                            }
                        )
                    else:
                        for res in ocr_results:
                            results.append(
                                {
                                    "track_id": track_id,
                                    "vehicle_class": vehicle_class,
                                    "crop_image": str(crop_path),
                                    "plate_image": str(plate_file),
                                    "text": res["text"],
                                    "confidence": res.get("confidence", 0.0),
                                    "method": res.get("method", "unknown"),
                                }
                            )

    save_results(outdir, results)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_file": str(outdir / "results.json"),
        "plates_dir": str(plates_dir),
        "num_entries": len(results),
        "used_easyocr": reader is not None,
        "used_tesseract": use_tesseract or reader is None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_plate_image(image: np.ndarray, path: Path) -> None:
    ensure_dirs(path.parent)
    if cv2 is not None and image.ndim == 2:
        cv2.imwrite(str(path), image)
    elif cv2 is not None and image.ndim == 3:
        cv2.imwrite(str(path), image)
    elif Image is not None:
        Image.fromarray(image).save(path)


def load_image(path: Path) -> Optional[np.ndarray]:
    if cv2 is not None:
        return cv2.imread(str(path))
    if Image is not None:
        return np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
    return None


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_results(outdir: Path, results: List[Dict[str, object]]) -> None:
    json_path = outdir / "results.json"
    csv_path = outdir / "results.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if results:
        headers = list(results[0].keys())
    else:
        headers = ["track_id", "vehicle_class", "crop_image", "plate_image", "text", "confidence", "method"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ANPR pipeline over vehicle crops")
    parser.add_argument("--crops-root", default=Path("outputs/crops"), type=Path)
    parser.add_argument("--outdir", default=Path("outputs/anpr"), type=Path)
    parser.add_argument("--vehicle-classes", nargs="*", help="Specific vehicle classes to scan")
    parser.add_argument("--no_preprocess", action="store_true", help="Disable plate preprocessing")
    parser.add_argument("--use_tesseract", action="store_true", help="Prefer pytesseract even if EasyOCR is available")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    process_crops(
        crops_root=args.crops_root,
        outdir=args.outdir,
        vehicle_classes=args.vehicle_classes,
        preprocess_enabled=not args.no_preprocess,
        use_tesseract=args.use_tesseract,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

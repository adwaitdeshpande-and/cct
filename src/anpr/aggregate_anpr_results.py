"""Aggregate per-track ANPR results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def aggregate(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in results:
        track_id = str(entry.get("track_id"))
        grouped[track_id].append(entry)

    aggregated: List[Dict[str, object]] = []
    for track_id, entries in grouped.items():
        stats: Dict[str, Dict[str, float]] = {}
        for entry in entries:
            text = entry.get("text", "")
            if not text:
                continue
            text = str(text)
            conf = float(entry.get("confidence", 0.0))
            if text not in stats:
                stats[text] = {"count": 0.0, "sum_conf": 0.0}
            stats[text]["count"] += 1
            stats[text]["sum_conf"] += conf
        best_text = ""
        best_score = -1.0
        for text, info in stats.items():
            count = info["count"]
            avg_conf = info["sum_conf"] / max(count, 1)
            score = count * avg_conf
            if score > best_score:
                best_score = score
                best_text = text
        aggregated.append(
            {
                "track_id": track_id,
                "best_plate": best_text,
                "score": best_score if best_text else 0.0,
                "num_candidates": len(entries),
            }
        )
    return aggregated


def process(results_path: Path, outdir: Path) -> List[Dict[str, object]]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    aggregated = aggregate(data)
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / "aggregated.json"
    csv_path = outdir / "aggregated.csv"
    json_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    headers = ["track_id", "best_plate", "score", "num_candidates"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)
    return aggregated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate ANPR results")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--outdir", default="outputs/anpr", help="Output directory for aggregated files")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    process(Path(args.results), Path(args.outdir))


if __name__ == "__main__":  # pragma: no cover
    main()

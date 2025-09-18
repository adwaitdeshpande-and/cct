import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from src.forensics.preprocess import preprocess


def test_preprocess_creates_manifest(tmp_path: Path):
    video_path = tmp_path / "sample.mp4"
    height, width = 32, 32
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height))
    for idx in range(5):
        frame = np.full((height, width, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    outdir = tmp_path / "out"
    manifest = preprocess(
        input_path=video_path,
        outdir=outdir,
        frame_skip=1,
        max_frames=3,
        use_exiftool=False,
        use_buftool=False,
    )

    manifest_path = outdir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["saved_frames"] <= 3
    frame_hashes = json.loads((outdir / "frame_hashes.json").read_text())
    assert len(frame_hashes) == data["saved_frames"]
    original_hash = (outdir / "original_sha256.txt").read_text().strip()
    assert len(original_hash) == 64

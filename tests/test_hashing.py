from pathlib import Path

from src.utils.hashing import sha256_bytes, sha256_file


def test_sha256_bytes():
    data = b"forensics"
    expected = "b83d7514ba17c3f1156a2648c1a9d3d167143e695ad491e6197f88441c7a1e4a"
    assert sha256_bytes(data) == expected


def test_sha256_file(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("forensics", encoding="utf-8")
    expected = "b83d7514ba17c3f1156a2648c1a9d3d167143e695ad491e6197f88441c7a1e4a"
    assert sha256_file(file_path) == expected

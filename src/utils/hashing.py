"""Utility helpers for computing SHA-256 digests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def _to_path(path: Union[str, Path]) -> Path:
    """Return a :class:`Path` object for *path*."""
    if isinstance(path, Path):
        return path
    return Path(path)


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest for the file located at *path*.

    The file is streamed in chunks so that arbitrarily large files can be
    processed without loading them into memory.
    """

    digest = hashlib.sha256()
    file_path = _to_path(path)
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for the provided *data* bytes."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_bytes expects a bytes-like object")
    return hashlib.sha256(bytes(data)).hexdigest()


__all__ = ["sha256_file", "sha256_bytes"]

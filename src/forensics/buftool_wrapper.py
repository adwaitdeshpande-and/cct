"""Wrapper utilities around the optional ``buftool`` command line utility."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Union


def run_buftool(path: Union[str, Path]) -> Dict[str, object]:
    """Execute ``buftool`` against *path* and return its JSON output.

    ``buftool`` is an optional dependency that some users employ for
    low-level container analysis.  The helper gracefully handles the
    executable not being present and returns a dictionary describing the
    failure so that the calling code can decide how to react.
    """

    executable = shutil.which("buftool")
    if executable is None:
        return {
            "error": "buftool_not_found",
            "message": "buftool executable is not available on PATH",
        }

    path = Path(path)
    try:
        result = subprocess.run(
            [executable, "-j", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - hard to simulate reliably in tests
        return {"error": "execution_failed", "message": str(exc)}

    if result.returncode != 0:
        return {
            "error": "buftool_failed",
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
        }

    output = result.stdout.strip()
    if not output:
        return {"error": "empty_output"}

    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        return {
            "error": "invalid_json",
            "message": str(exc),
            "raw_output": output,
        }


__all__ = ["run_buftool"]

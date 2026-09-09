from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_module_cli_has_help_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "native_rag.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        },
    )
    assert "build a document index" in result.stdout
    assert "run a minimal model-only forward/decode" in result.stdout

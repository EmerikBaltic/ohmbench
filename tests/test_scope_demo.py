"""Tests for offscreen oscilloscope PNG demo exporter (`circuitsim.tools.scope_demo`)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from PySide6.QtGui import QColor, QImage

from circuitsim.tools.scope_demo import export_demo_pngs


def test_export_demo_pngs_creates_distinct_non_blank_images(tmp_path: Path) -> None:
    """All 4 demo PNGs save cleanly with distinct non-uniform waveform pixels."""
    paths = export_demo_pngs(tmp_path)
    expected_names = {
        "scope_sine.png",
        "scope_square.png",
        "scope_rc_step.png",
        "scope_fft.png",
    }
    assert {p.name for p in paths} == expected_names

    bg_rgb = QColor("#0B0F14").rgb()
    for p in paths:
        assert p.exists() and p.stat().st_size > 2000
        img = QImage(str(p))
        assert not img.isNull()
        assert img.width() == 960 and img.height() == 540
        # Verify waveform area is not blank background
        non_bg = 0
        for x in range(80, 900, 12):
            for y in range(60, 460, 12):
                if img.pixel(x, y) != bg_rgb:
                    non_bg += 1
        assert non_bg > 20


def test_scope_demo_cli_offscreen_subprocess(tmp_path: Path) -> None:
    """CLI `--export` runs cleanly in an isolated offscreen subprocess."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuitsim.tools.scope_demo",
            "--export",
            str(tmp_path),
        ],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    for name in ("scope_sine.png", "scope_square.png", "scope_rc_step.png", "scope_fft.png"):
        assert (tmp_path / name).exists()

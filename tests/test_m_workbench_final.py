"""Phase M regression suite: DSO polish, Function Generator, AC RMS meters, Logic Pack, Quick-Place & Custom Parts Library."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_m0_phase_m_scaffolding(qapp: QApplication) -> None:
    assert qapp is not None

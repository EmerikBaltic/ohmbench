"""Phase H Workbench Expansion tests (H0-H10) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary


def test_h0_phase_h_scaffolding(qapp) -> None:
    """Verify offscreen Qt scaffolding and empty CircuitScene setup."""
    assert qapp is not None
    scene = CircuitScene(SymbolLibrary())
    try:
        assert scene.placed_items == []
        assert scene.placed_wires == []
        assert scene.wire_graph.wires == ()
    finally:
        scene.shutdown_scene()
        qapp.processEvents()

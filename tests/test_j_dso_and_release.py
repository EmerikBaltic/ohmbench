"""Phase J Benchtop DSO-1062D, SPICE Exporter, Docs & Windows Packaging tests (J1-J6)."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine import MNASolver, SimulationSession
from circuitsim.engine.netlist import Circuit
from circuitsim.wiring.model import TerminalRef


def test_j1_phase_j_scaffolding(qapp) -> None:
    """J1: Verify offscreen Qt scaffolding and empty CircuitScene setup."""
    assert qapp is not None
    scene = CircuitScene(SymbolLibrary())
    try:
        assert scene.placed_items == []
        assert scene.wire_graph.wires == ()
    finally:
        scene.shutdown_scene()
        qapp.processEvents()

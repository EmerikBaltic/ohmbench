"""Phase I Advanced Workbench tests (I0-I10) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary


def test_i0_phase_i_scaffolding(qapp) -> None:
    """I0: Verify offscreen Qt scaffolding and empty CircuitScene setup."""
    assert qapp is not None
    scene = CircuitScene(SymbolLibrary())
    try:
        assert scene.placed_items == []
        assert scene.placed_wires == []
        assert scene.wire_graph.wires == ()
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_i1_full_span_collision_blocks_overlap_allows_series_and_reverts_drag(qapp) -> None:
    """I1: Full-span footprint blocks overlaps/half-overlaps, allows end-to-end series, and reverts colliding drags."""
    scene = CircuitScene(SymbolLibrary())
    try:
        r1 = scene.place_new_component("resistor", QPointF(0.0, 0.0))
        assert r1 is not None
        assert len(scene.placed_items) == 1

        # All 5 overlapping/half-overlapping positions must be rejected without spawning components
        for bad_pt in (
            QPointF(0.0, 0.0),
            QPointF(20.0, 0.0),
            QPointF(40.0, 0.0),
            QPointF(-40.0, 0.0),
            QPointF(0.0, 20.0),
        ):
            assert scene.place_new_component("resistor", bad_pt) is None
            assert len(scene.placed_items) == 1
            assert "Cannot place component" in scene.last_status_notice

        # End-to-end series placement at (80, 0) touches terminal cleanly at (40, 0) without collision
        r2 = scene.place_new_component("resistor", QPointF(80.0, 0.0))
        assert r2 is not None
        assert len(scene.placed_items) == 2

        # Dragging R3 from (0, 120) onto half-overlapping (40, 0) must revert to (0, 120)
        r3 = scene.place_new_component("resistor", QPointF(0.0, 120.0))
        assert r3 is not None
        scene.begin_component_move_tracking()
        r3.setPos(QPointF(40.0, 0.0))
        assert scene.commit_component_move_tracking() is False
        assert (r3.pos().x(), r3.pos().y()) == (0.0, 120.0)
    finally:
        scene.shutdown_scene()
        qapp.processEvents()

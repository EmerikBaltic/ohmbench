"""Phase I Advanced Workbench tests (I0-I10) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine import SimulationSession
from circuitsim.wiring.model import TerminalRef


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


def test_i2_overload_badge_populates_problems_panel_tooltip_and_inspector(qapp, tmp_path: Path) -> None:
    """I2: Component stress overload populates Problems panel, hover tooltip, and inspector banner."""
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "i2_settings"))
    try:
        scene = win.circuit_scene
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, 0.0))
        gnd = scene.place_new_component("ground", QPointF(0.0, 140.0))
        assert v1 is not None and r1 is not None and gnd is not None

        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        # 12 V across 500 ohm (0.25 W rated) -> P = 144 / 500 = 288.0 mW (115.2% OVERLOAD)
        scene.edit_component_parameters(r1, {"resistance": 500.0})
        scene.add_wire(TerminalRef(v1.component.name, "pos"), TerminalRef(r1.component.name, "a"))
        scene.add_wire(TerminalRef(r1.component.name, "b"), TerminalRef(gnd.component.name, "gnd"))
        scene.add_wire(TerminalRef(v1.component.name, "neg"), TerminalRef(gnd.component.name, "gnd"))

        frame = SimulationSession(connectivity=scene.extract_connectivity_result()).start()
        win._on_simulation_frame(frame)
        qapp.processEvents()

        # (a) Problems panel lists R1 overload mentioning 288 mW and 250 mW
        tree = win.problems_dock.panel.tree
        assert tree.topLevelItemCount() >= 1
        r1_rows = [
            tree.topLevelItem(i)
            for i in range(tree.topLevelItemCount())
            if "R1" in tree.topLevelItem(i).text(2)
        ]
        assert r1_rows
        assert "288" in r1_rows[0].text(1) and "250" in r1_rows[0].text(1)

        # (b) Component tooltip contains 288 and overload
        tip = r1.toolTip().lower()
        assert "288" in tip and "overload" in tip

        # (c) Selecting R1 shows overload warning in Properties / Info panel
        scene.clearSelection()
        r1.setSelected(True)
        qapp.processEvents()
        info_txt = win.info_label.text().lower()
        assert "overload" in info_txt and "288" in info_txt

        # (d) Raising R1 to 1000 ohm (144 mW < 250 mW) clears Problems panel back to 0
        scene.edit_component_parameters(r1, {"resistance": 1_000.0})
        frame2 = SimulationSession(connectivity=scene.extract_connectivity_result()).start()
        win._on_simulation_frame(frame2)
        qapp.processEvents()
        assert tree.topLevelItemCount() == 0
        assert win.problems_dock.panel.count_label.text() == "(0)"
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()

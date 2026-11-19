"""Phase I Advanced Workbench tests (I0-I10) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import numpy as np
import pytest

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


def test_i3_light_theme_contrast_dock_title_and_synced_shortcuts_help(qapp, tmp_path, monkeypatch) -> None:
    """I3: Balanced matte Light Theme contrast, QDockWidget::title light styling, and dynamic shortcuts help."""
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.themes import LIGHT_PALETTE, ThemeMode, build_stylesheet, get_theme_palette

    monkeypatch.setenv("OHMBENCH_DATA_DIR", str(tmp_path))

    # (a) Distinct non-glaring Light Theme tokens and styled QDockWidget::title
    assert LIGHT_PALETTE.window_bg.upper() != "#FFFFFF"
    assert LIGHT_PALETTE.panel_bg.upper() != "#FFFFFF"
    assert LIGHT_PALETTE.canvas_bg.upper() != "#FFFFFF"
    assert len(
        {
            LIGHT_PALETTE.window_bg.upper(),
            LIGHT_PALETTE.panel_bg.upper(),
            LIGHT_PALETTE.canvas_bg.upper(),
        }
    ) == 3
    assert LIGHT_PALETTE.grid_minor.upper() != LIGHT_PALETTE.grid_major.upper()

    light_qss = build_stylesheet(LIGHT_PALETTE)
    assert "QDockWidget::title" in light_qss
    assert "#DCE2EA" in light_qss
    assert "#1E1E1E" not in light_qss

    # (b) Properties / Info panel shortcut text matches AppSettings.shortcuts dynamically
    win = MainWindow()
    try:
        win.apply_theme(ThemeMode.LIGHT)
        assert win.circuit_scene.theme == ThemeMode.LIGHT
        assert get_theme_palette(win.circuit_scene.theme) is LIGHT_PALETTE
        assert "#DCE2EA" in win.styleSheet()
        default_help = win.info_label.text()
        assert "Rotate 90° CW: Ctrl+R" in default_help
        assert "Rotate 90° CW: R\n" not in default_help
        assert "Resistor: R" in default_help

        win.settings.set_shortcut("rotate", "Alt+R")
        win.apply_shortcuts()
        qapp.processEvents()
        updated_help = win.info_label.text()
        assert "Rotate 90° CW: Alt+R" in updated_help
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()


def test_i4_ne555_dc_states_astable_oscillation_and_comparator() -> None:
    """I4: NE555 DC latch/discharge states, astable RC oscillation (3V..6V), and Comparator switching."""
    from circuitsim.components import (
        Capacitor,
        Comparator,
        DCVoltageSource,
        Ground,
        NE555,
        Resistor,
        create_default_component_model,
    )
    from circuitsim.engine.netlist import Circuit
    from circuitsim.engine.solver import MNASolver

    # Factory round-trip & G1 parameter validation
    assert create_default_component_model("ne555", "U1").component_type == "ne555"
    assert create_default_component_model("comparator", "U2").component_type == "comparator"

    # (a) 555 Timer DC latch & open-collector discharge test
    circ_dc = Circuit()
    vcc = DCVoltageSource("V_CC", "vcc", "0", voltage=9.0)
    vtrig = DCVoltageSource("V_TRIG", "trig", "0", voltage=2.0)
    vthr = DCVoltageSource("V_THR", "thr", "0", voltage=2.0)
    vrst = DCVoltageSource("V_RST", "rst", "0", voltage=9.0)
    r_pull = Resistor("R_PULL", "vcc", "dis", resistance=10_000.0)
    u555 = NE555("U1", "vcc", "0", "trig", "thr", "dis", "out", "rst")
    gnd = Ground("GND1", "0")
    for c in (vcc, vtrig, vthr, vrst, r_pull, u555, gnd):
        circ_dc.add(c)

    solver_dc = MNASolver(circ_dc)
    res1 = solver_dc.solve_dc()
    assert res1.node_voltages["out"] == pytest.approx(9.0, abs=0.2)
    assert res1.node_voltages["dis"] > 8.9  # DIS high-Z -> pulled to 9V

    # Hold inside [1/3 Vcc, 2/3 Vcc] = [3V, 6V]
    vtrig.dc_voltage = 4.5
    vthr.dc_voltage = 4.5
    res2 = solver_dc.solve_dc()
    assert res2.node_voltages["out"] == pytest.approx(9.0, abs=0.2)

    # Threshold > 6V -> OUT ~ 0V, DIS pulls 10k pull-up near 0V
    vthr.dc_voltage = 7.0
    res3 = solver_dc.solve_dc()
    assert res3.node_voltages["out"] < 0.1
    assert res3.node_voltages["dis"] < 0.05

    # Low RESET (< 0.7V) overrides TRIG (< 3V) and forces OUT low
    vtrig.dc_voltage = 2.0
    vrst.dc_voltage = 0.2
    res4 = solver_dc.solve_dc()
    assert res4.node_voltages["out"] < 0.1

    # (b) 555 Astable oscillator transient: VCC=9V, RA=1k, RB=2k, C=1uF -> oscillates across [3V, 6V]
    circ_ast = Circuit()
    circ_ast.add(DCVoltageSource("V1", "vcc", "0", voltage=9.0))
    circ_ast.add(Ground("G1", "0"))
    circ_ast.add(Resistor("R_A", "vcc", "dis", resistance=1_000.0))
    circ_ast.add(Resistor("R_B", "dis", "cap", resistance=2_000.0))
    circ_ast.add(Capacitor("C1", "cap", "0", capacitance=1e-6))
    circ_ast.add(NE555("U1", "vcc", "0", "cap", "cap", "dis", "out", "vcc"))

    tr = MNASolver(circ_ast).solve_transient(duration=6.5e-3, dt=1e-5)
    v_cap = tr.node_voltages["cap"]
    v_out = tr.node_voltages["out"]
    assert float(v_out.max()) > 8.8
    assert float(v_out.min()) < 0.2
    assert float(v_cap.max()) >= 5.95
    # Verify cap climbed to ~6V then discharged back down toward 3V
    idx_peak = int(v_cap.argmax())
    assert idx_peak < len(v_cap) - 10
    assert float(v_cap[idx_peak:].min()) <= 3.1

    # (c) Voltage Comparator switching between v_low=0V and v_high=5V
    circ_cmp = Circuit()
    vp = DCVoltageSource("VP", "np", "0", voltage=3.0)
    vn = DCVoltageSource("VN", "nn", "0", voltage=2.5)
    cmp_ic = Comparator("U_CMP", "np", "nn", "out", v_high=5.0, v_low=0.0)
    circ_cmp.add(vp)
    circ_cmp.add(vn)
    circ_cmp.add(cmp_ic)
    circ_cmp.add(Resistor("R_LOAD", "out", "0", resistance=10_000.0))
    circ_cmp.add(Ground("G_CMP", "0"))

    cmp_solver = MNASolver(circ_cmp)
    r_hi = cmp_solver.solve_dc()
    assert r_hi.node_voltages["out"] == pytest.approx(5.0, abs=0.05)

    vp.dc_voltage = 2.0
    r_lo = cmp_solver.solve_dc()
    assert r_lo.node_voltages["out"] == pytest.approx(0.0, abs=0.05)

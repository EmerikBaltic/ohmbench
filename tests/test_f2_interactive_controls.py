"""Tests for Item F2 + G1 + G2: Switch/Potentiometer interactive controls without inspector QSlider."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QSlider

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.canvas.view import CircuitView
from circuitsim.components import (
    DCVoltageSource,
    Ground,
    Potentiometer,
    Resistor,
    Switch,
)
from circuitsim.engine import Circuit, MNASolver, SimulationSession
from circuitsim.wiring import TerminalRef, WireGraph, build_circuit as wiring_build_circuit
from tests.ui_flow_helpers import click_viewport, scene_to_viewport_point


def test_f2_potentiometer_qt_free_physics_and_parameter_metadata() -> None:
    """F2 + G1: ComponentParameter accepts optional min/max/description; Potentiometer divides cleanly."""
    pot = Potentiometer("POT1", "a", "w", "b", resistance=10_000.0, alpha=0.5)
    params = pot.get_editable_parameters()
    assert [p.key for p in params] == ["resistance", "alpha"]
    assert pot.MIN_ALPHA == 0.001 and pot.MAX_ALPHA == 0.999
    assert math.isclose(Potentiometer("P0", "a", "w", "b", alpha=0.0).alpha, 0.001, rel_tol=1e-12)
    assert math.isclose(Potentiometer("P1", "a", "w", "b", alpha=1.0).alpha, 0.999, rel_tol=1e-12)

    v1 = DCVoltageSource("V1", "n_pos", "0", voltage=10.0)
    gnd = Ground("GND1", "0")
    pot_wire = Potentiometer("POT1", "n_pos", "n_w", "0", resistance=10_000.0, alpha=0.25)
    circuit = Circuit()
    for comp in (v1, pot_wire, gnd):
        circuit.add(comp)
    for alpha_val, exp_v in ((0.25, 2.500), (0.50, 5.000), (0.75, 7.500)):
        pot_wire.set_alpha(alpha_val)
        circuit.mark_modified()
        res = MNASolver(circuit).solve_dc()
        assert math.isclose(res.node_voltages["n_w"], exp_v, rel_tol=1e-6, abs_tol=1e-6)

    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("POT1", "a"))
    g = g.add_wire(TerminalRef("V1", "neg"), TerminalRef("POT1", "b"))
    g = g.add_wire(TerminalRef("POT1", "b"), TerminalRef("GND1", "gnd"))
    wnet = wiring_build_circuit(g, [v1, pot_wire, gnd])
    wnet.circuit.get_component("POT1").set_alpha(0.25)  # type: ignore[union-attr]
    res2 = MNASolver(wnet.circuit).solve_dc()
    w_node = wnet.component_nodes["POT1"]["w"]
    assert math.isclose(res2.node_voltages[w_node], 2.500, rel_tol=1e-6, abs_tol=1e-6)


def test_f2_interactive_switch_click_toggles_live_current(qapp) -> None:
    """F2: Clicking an interactive SPST switch on CircuitView toggles 0.0 mA <-> 24.0 mA on 12V + 500 Ohm."""
    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    view.resize(900, 600)
    view.show()
    view.set_view_state(1.0, (0.0, 0.0))
    qapp.processEvents()

    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-200.0, 0.0))
        sw1 = scene.place_new_component("switch", QPointF(-60.0, 0.0))
        r1 = scene.place_new_component("resistor", QPointF(80.0, 0.0))
        gnd = scene.place_new_component("ground", QPointF(-200.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 500.0})

        for a, b in (
            (TerminalRef("V1", "pos"), TerminalRef("SW1", "a")),
            (TerminalRef("SW1", "b"), TerminalRef("R1", "a")),
            (TerminalRef("R1", "b"), TerminalRef("V1", "neg")),
            (TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd")),
        ):
            scene.add_wire(a, b)

        conn = scene.extract_connectivity_result()
        session = SimulationSession(connectivity=conn)
        frame = session.start()
        assert abs(frame.components["R1"].current) < 1e-6  # Open switch (~0.0 mA)

        def _on_sw_toggle(name: str, _closed: bool) -> None:
            nonlocal frame
            frame = session.toggle_switch(name)

        scene.switch_toggled_by_user.connect(_on_sw_toggle)

        sw_vp = scene_to_viewport_point(view, sw1.pos().x(), sw1.pos().y())
        click_viewport(view, sw_vp)
        assert sw1.component.closed is True
        assert math.isclose(abs(frame.components["R1"].current), 0.024, rel_tol=1e-2, abs_tol=1e-4)

        click_viewport(view, sw_vp)
        assert sw1.component.closed is False
        assert abs(frame.components["R1"].current) < 1e-6

        # Momentary hold & release
        sw1.component.momentary = True
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, sw_vp)
        qapp.processEvents()
        assert sw1.component.closed is True
        assert math.isclose(abs(frame.components["R1"].current), 0.024, rel_tol=1e-2, abs_tol=1e-4)

        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, sw_vp)
        qapp.processEvents()
        assert sw1.component.closed is False
        assert abs(frame.components["R1"].current) < 1e-6
    finally:
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()


def test_f2_potentiometer_alpha_steps_single_undo_and_no_inspector_slider(tmp_path, qapp) -> None:
    """F2 + G2: Potentiometer alpha edits yield 2.5/5.0/7.5 V with 1-step undo and no QSlider in inspector."""
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "f2_g2"))
    win.resize(1000, 720)
    win.show()
    qapp.processEvents()
    scene = win.circuit_scene

    try:
        assert win.info_dock.findChildren(QSlider) == []

        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0))
        pot = scene.place_new_component("potentiometer", QPointF(40.0, 0.0))
        gnd = scene.place_new_component("ground", QPointF(40.0, 140.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 10.0})

        for a, b in (
            (TerminalRef("V1", "pos"), TerminalRef("POT1", "a")),
            (TerminalRef("POT1", "b"), TerminalRef("GND1", "gnd")),
            (TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd")),
        ):
            scene.add_wire(a, b)

        scene.clearSelection()
        pot.setSelected(True)
        qapp.processEvents()
        assert win.info_dock.findChildren(QSlider) == []

        for alpha_val, exp_vw in ((0.25, 2.500), (0.50, 5.000), (0.75, 7.500)):
            undo_before = scene.undo_stack.count()
            scene.edit_component_parameters(pot, {"alpha": alpha_val})
            assert scene.undo_stack.count() == undo_before + 1
            res = MNASolver(scene.build_circuit()).solve_dc()
            wnet = scene.extract_connectivity_result()
            w_node = wnet.component_nodes["POT1"]["w"]
            assert math.isclose(res.node_voltages[w_node], exp_vw, rel_tol=1e-6, abs_tol=1e-6)

        # Single-step undo returns from 0.75 back to 0.50 -> 5.000 V
        scene.undo_stack.undo()
        assert math.isclose(pot.component.alpha, 0.50, rel_tol=1e-9)
        res_undo = MNASolver(scene.build_circuit()).solve_dc()
        w_node_undo = scene.extract_connectivity_result().component_nodes["POT1"]["w"]
        assert math.isclose(res_undo.node_voltages[w_node_undo], 5.000, rel_tol=1e-6, abs_tol=1e-6)
    finally:
        win._is_closing = True
        win.close()
        qapp.processEvents()

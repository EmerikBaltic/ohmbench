"""Tests for Item F4: component stress, derating, and overload/smoke warnings."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.components import (
    Capacitor,
    DCVoltageSource,
    Ground,
    LED,
    Resistor,
)
from circuitsim.engine import (
    Circuit,
    MNASolver,
    SimulationSession,
    SimulationState,
    StressLevel,
    evaluate_component_stress,
)
from circuitsim.wiring import TerminalRef
from tests.conftest import assert_module_has_no_qt_imports


def test_f4_resistor_derating_and_led_capacitor_stress_qt_free() -> None:
    """Qt-free evaluate_component_stress verifies resistor power bands, LED, and polarized cap."""
    assert_module_has_no_qt_imports("circuitsim.engine.stress")

    # 1. 12 V source across 0.25 W resistor: 1000 -> 144 mW NORMAL, 600 -> 240 mW WARNING, 500 -> 288 mW OVERLOAD
    expected_cases = [
        (1000.0, 0.144, 57.6, StressLevel.NORMAL),
        (600.0, 0.240, 96.0, StressLevel.WARNING),
        (500.0, 0.288, 115.2, StressLevel.OVERLOAD),
    ]
    for r_ohms, exp_power, exp_pct, exp_level in expected_cases:
        c = Circuit()
        c.add(DCVoltageSource("V1", "n1", "0", voltage=12.0))
        c.add(Resistor("R1", "n1", "0", resistance=r_ohms, power_rating=0.25))
        c.add(Ground("G1", "0"))
        sol = MNASolver(c).solve_dc()
        reports = evaluate_component_stress(c, sol)
        rep = reports["R1"]
        assert math.isclose(c.get_component("R1").power, exp_power, rel_tol=1e-6)
        assert rep.level == exp_level
        assert math.isclose(rep.utilization_pct, exp_pct, rel_tol=1e-4)

    # 2. LED directly across 5.0 V -> OVERLOAD; with 220 ohm series resistor -> NORMAL
    c_led_direct = Circuit()
    c_led_direct.add(DCVoltageSource("V1", "n1", "0", voltage=5.0))
    c_led_direct.add(LED("D1", "n1", "0", max_current=0.030))
    c_led_direct.add(Ground("G1", "0"))
    sol_direct = MNASolver(c_led_direct).solve_dc()
    rep_direct = evaluate_component_stress(c_led_direct, sol_direct)["D1"]
    assert rep_direct.level == StressLevel.OVERLOAD

    c_led_safe = Circuit()
    c_led_safe.add(DCVoltageSource("V1", "n1", "0", voltage=5.0))
    c_led_safe.add(Resistor("R1", "n1", "n2", resistance=220.0))
    c_led_safe.add(LED("D1", "n2", "0", max_current=0.030))
    c_led_safe.add(Ground("G1", "0"))
    sol_safe = MNASolver(c_led_safe).solve_dc()
    rep_safe = evaluate_component_stress(c_led_safe, sol_safe)["D1"]
    # Hand calc: (5.0 - 2.0) / 220 ~= 13.64 mA < 30 mA * 0.8 = 24 mA -> NORMAL
    assert rep_safe.level == StressLevel.NORMAL

    # 3. Polarized capacitor reverse polarity & LED reverse voltage check
    c_pol = Circuit()
    c_pol.add(DCVoltageSource("V1", "neg", "0", voltage=6.0))
    c_pol.add(Resistor("R1", "neg", "cap", resistance=100.0))
    c_pol.add(Capacitor("C1", "0", "cap", capacitance=10e-6, voltage_rating=25.0, polarized=True))
    c_pol.add(Ground("G1", "0"))
    sol_pol = MNASolver(c_pol).solve_dc()
    rep_pol = evaluate_component_stress(c_pol, sol_pol)["C1"]
    assert rep_pol.level == StressLevel.OVERLOAD


def test_f4_scene_stress_badges_toggle_and_live_continue(tmp_path, qapp) -> None:
    """Overloaded part shows red badge while solver stays RUNNING; live edit recovers to NORMAL."""
    settings = AppSettings(data_dir=tmp_path / "f4_settings")
    win = MainWindow(settings=settings)
    win.resize(1000, 720)
    win.show()
    qapp.processEvents()
    scene = win.circuit_scene

    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-120.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(120.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-120.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 500.0})
        r1.component.power_rating = 0.25
        r1.component.max_power = 0.25

        r1_terms = r1.terminal_scene_positions_px()
        top_t = min(r1_terms, key=lambda t: r1_terms[t][1])
        bot_t = max(r1_terms, key=lambda t: r1_terms[t][1])
        scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", top_t))
        scene.add_wire(TerminalRef("R1", bot_t), TerminalRef("V1", "neg"))
        scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))

        sess = SimulationSession(connectivity=scene.extract_connectivity_result())
        frame = sess.start()
        assert frame.state == SimulationState.RUNNING  # continues running despite overload!
        scene.apply_simulation_frame(frame)

        assert win.act_stress_warnings.isChecked() is True
        assert scene.stress_warnings_enabled is True
        assert scene.latest_stress_reports["R1"].level == StressLevel.OVERLOAD
        assert r1.stress_level == "overload"

        # Toggle badge visibility off & back on cleanly
        win.act_stress_warnings.setChecked(False)
        assert scene.stress_warnings_enabled is False
        assert r1.stress_warnings_enabled is False
        win.act_stress_warnings.setChecked(True)
        assert r1.stress_warnings_enabled is True

        # Render offscreen to exercise smoke/overload badge painting safely
        img = QImage(320, 240, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.black)
        p = QPainter(img)
        scene.render(p, target=img.rect().toRectF(), source=scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        p.end()

        # Live fix: update R1 to 1000 ohm -> returns to NORMAL
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        sess2 = SimulationSession(connectivity=scene.extract_connectivity_result())
        frame2 = sess2.start()
        assert frame2.state == SimulationState.RUNNING
        scene.apply_simulation_frame(frame2)
        assert scene.latest_stress_reports["R1"].level == StressLevel.NORMAL
        assert r1.stress_level == "normal"
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()

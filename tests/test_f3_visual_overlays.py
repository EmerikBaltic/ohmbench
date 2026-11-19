"""Tests for Item F3: voltage color-coding, animated current dots, and Alt-hover power probe."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.engine import (
    SimulationSession,
    compute_current_dot_velocity,
    map_voltage_to_rgb,
)
from circuitsim.wiring import (
    TerminalRef,
    compute_current_dot_velocity as wiring_dot_velocity,
    map_voltage_to_rgb as wiring_voltage_rgb,
)
from tests.conftest import assert_module_has_no_qt_imports


def test_f3_qt_free_voltage_rgb_and_current_velocity() -> None:
    """Qt-free voltage RGB interpolation and current dot velocity helpers work cleanly."""
    assert_module_has_no_qt_imports(
        "circuitsim.wiring", "circuitsim.wiring.overlay", "circuitsim.engine"
    )
    assert map_voltage_to_rgb is wiring_voltage_rgb
    assert compute_current_dot_velocity is wiring_dot_velocity

    # 10 V / (1k + 2k) divider node voltages: +10.0 V, +6.667 V, 0.0 V, and -10.0 V
    rgb_zero = map_voltage_to_rgb(0.0, v_ref=12.0)
    rgb_mid = map_voltage_to_rgb(20.0 / 3.0, v_ref=12.0)
    rgb_pos = map_voltage_to_rgb(10.0, v_ref=12.0)
    rgb_neg = map_voltage_to_rgb(-10.0, v_ref=12.0)

    # Neutral gray at 0.0 V
    assert rgb_zero[0] == rgb_zero[1] == rgb_zero[2]
    # Positive voltages shift green monotonically
    assert rgb_mid[1] > rgb_mid[0] and rgb_mid[1] > rgb_mid[2]
    assert rgb_pos[1] > rgb_mid[1] > rgb_zero[1]
    assert rgb_pos[0] < rgb_mid[0] < rgb_zero[0]
    # Negative voltages shift red
    assert rgb_neg[0] > rgb_neg[1] and rgb_neg[0] > rgb_neg[2]
    assert rgb_neg[0] > rgb_zero[0]

    # Hand calc: I = 10.0 V / 3000.0 Ohm = 3.3333 mA -> normalized velocity > 0
    i_div = 10.0 / 3000.0
    v_dot = compute_current_dot_velocity(i_div, i_scale=0.05)
    assert math.isclose(v_dot, i_div / 0.05, rel_tol=1e-9)
    assert v_dot > 0.0
    assert math.isclose(compute_current_dot_velocity(-i_div, i_scale=0.05), -v_dot, rel_tol=1e-9)
    assert compute_current_dot_velocity(5e-10) == 0.0
    assert compute_current_dot_velocity(-5e-10) == 0.0


def test_f3_divider_wire_overlays_animation_and_alt_power_probe(tmp_path, qapp) -> None:
    """Voltage colors, current dots, view menu toggles, and Alt-hover power probe (11.11 / 22.22 / 33.33 mW)."""
    settings = AppSettings(data_dir=tmp_path / "f3_settings")
    win = MainWindow(settings=settings)
    win.resize(1100, 800)
    win.show()
    qapp.processEvents()
    view = win.circuit_view
    scene = win.circuit_scene
    view.set_view_state(1.0, (0.0, 0.0))

    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -40.0))
        r2 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})

        w_top = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
        w_mid = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("R2", "a"))
        w_bot = scene.add_wire(TerminalRef("R2", "b"), TerminalRef("V1", "neg"))
        w_gnd = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
        assert all(w is not None for w in (w_top, w_mid, w_bot, w_gnd))

        # Toggle View menu actions on
        assert not scene.voltage_colors_enabled
        assert not scene.current_dots_enabled
        win.act_voltage_colors.setChecked(True)
        win.act_current_dots.setChecked(True)
        assert scene.voltage_colors_enabled is True
        assert scene.current_dots_enabled is True

        sess = SimulationSession(connectivity=scene.extract_connectivity_result())
        frame = sess.start()
        scene.apply_simulation_frame(frame)
        qapp.processEvents()

        assert w_top.voltage_color_enabled is True
        assert w_mid.voltage_color_enabled is True
        assert w_top.overlay_rgb == map_voltage_to_rgb(10.0)
        assert w_mid.overlay_rgb == map_voltage_to_rgb(20.0 / 3.0)
        assert w_bot.overlay_rgb == map_voltage_to_rgb(0.0)
        assert w_gnd.overlay_rgb == map_voltage_to_rgb(0.0)

        # Signed current flow V1.pos -> R1.a (+3.333 mA), R1.b -> R2.a (+3.333 mA), GND -> 0 A
        expected_vel = compute_current_dot_velocity(10.0 / 3000.0)
        assert math.isclose(w_top.dot_velocity, expected_vel, rel_tol=1e-5)
        assert math.isclose(w_mid.dot_velocity, expected_vel, rel_tol=1e-5)
        assert math.isclose(w_bot.dot_velocity, expected_vel, rel_tol=1e-5)
        assert w_gnd.dot_velocity == 0.0

        phase_0 = w_top.dot_phase
        scene.step_current_dot_animation(step_scale=0.5)
        assert w_top.dot_phase != phase_0

        # Render canvas offscreen cleanly with overlays enabled (no paint errors)
        img = QImage(480, 320, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.black)
        painter = QPainter(img)
        scene.render(painter, target=img.rect().toRectF(), source=scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        painter.end()

        # Alt-hover power thermometer probe via real viewport move + direct helper:
        # Hand calc: R1 -> 11.11 mW, R2 -> 22.22 mW, V1 total supplied -> 33.33 mW
        vp = view.viewport()
        QTest.mouseMove(vp, view.mapFromScene(r1.pos()))
        qapp.processEvents()
        p_r1 = scene.query_alt_power_probe_at(r1.pos())
        p_r2 = scene.query_alt_power_probe_at(r2.pos())
        p_v1 = scene.query_alt_power_probe_at(v1.pos())
        assert p_r1 is not None and "11.11 mW" in p_r1
        assert p_r2 is not None and "22.22 mW" in p_r2
        assert p_v1 is not None and "33.33 mW" in p_v1
        assert math.isclose(r1.component.power + r2.component.power, abs(v1.component.power), rel_tol=1e-6)
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()

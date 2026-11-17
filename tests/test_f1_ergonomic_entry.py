"""Tests for Item F1: direct hotkeys, Pin-0 polarity markers, and tabbed parameter dialog."""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QPointF, Qt

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.component_item import ComponentItem
from circuitsim.canvas.symbol_geometry import passive_pin_markers
from circuitsim.components import ACVoltageSource, DCVoltageSource
from circuitsim.dialogs.quick_edit_dialog import QuickValueDialog
from circuitsim.dialogs.value_parser import analyze_engineering_input, parse_engineering_value
from circuitsim.engine import MNASolver, SimulationSession
from circuitsim.wiring.model import TerminalRef
from tests.ui_flow_helpers import (
    click_viewport,
    create_window_and_view,
    draw_wire_click_to_click,
    press_key,
    scene_to_viewport_point,
    terminal_scene_point,
)
from tests.workbench_helpers import open_component_dialog, trigger_placement_hotkey


def test_f1_engineering_parser_distinguishes_m_and_meg_and_warns() -> None:
    """F1(c): parser distinguishes m (1e-3) from Meg (1e6) and warns on ambiguous units."""
    assert math.isclose(parse_engineering_value("4.7k"), 4700.0, rel_tol=1e-12)
    assert math.isclose(parse_engineering_value("100n"), 1e-7, rel_tol=1e-12)
    assert math.isclose(parse_engineering_value("1m"), 1e-3, rel_tol=1e-12)
    assert math.isclose(parse_engineering_value("1Meg"), 1e6, rel_tol=1e-12)
    assert math.isclose(parse_engineering_value("2.5G"), 2.5e9, rel_tol=1e-12)

    val_mhz, warn_mhz, err_mhz = analyze_engineering_input("1mhz")
    assert err_mhz is None and math.isclose(val_mhz or 0.0, 1e-3, rel_tol=1e-12)
    assert warn_mhz is not None and "millihertz" in warn_mhz

    val_M, warn_M, err_M = analyze_engineering_input("10M")
    assert err_M is None and math.isclose(val_M or 0.0, 1e7, rel_tol=1e-12)
    assert warn_M is not None and "Mega" in warn_M

    val_meg, warn_meg, err_meg = analyze_engineering_input("10Meg")
    assert err_meg is None and warn_meg is None and math.isclose(val_meg or 0.0, 1e7, rel_tol=1e-12)


def test_f1_all_hotkeys_menus_toolbar_and_dialog_tabs(tmp_path, qapp) -> None:
    """F1(a)(c): single-key hotkeys, visible menus/toolbar actions, source mode tabs, SPICE preview & ratings."""
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "f1_ui"))
    win.resize(1100, 800)
    win.show()
    qapp.processEvents()
    view = win.circuit_view
    scene = win.circuit_scene
    view.set_view_state(1.0, (0.0, 0.0))
    qapp.processEvents()

    try:
        # Visible menus & toolbar actions exist for every placement hotkey + Rotate (Ctrl+R)
        assert win.place_menu is not None
        for cid in ("resistor", "capacitor", "inductor", "led", "dc_voltage", "dc_current", "ground"):
            assert cid in win.place_actions
            assert win.place_actions[cid].isVisible()
        assert "Ctrl+R" in win.act_rotate.shortcut().toString()

        # Hotkeys spawn/activate placement cleanly: C, L, D, I, G, Ctrl+R rotate ghost, Esc cancels
        for key, expected_type, drop_pt in (
            (Qt.Key.Key_C, "capacitor", (-200.0, -100.0)),
            (Qt.Key.Key_L, "inductor", (-80.0, -100.0)),
            (Qt.Key.Key_D, "led", (40.0, -100.0)),
            (Qt.Key.Key_I, "dc_current", (160.0, -100.0)),
            (Qt.Key.Key_G, "ground", (-200.0, 100.0)),
        ):
            trigger_placement_hotkey(view, key, scene_pos=drop_pt)
            assert view.placing_component_type == expected_type
            click_viewport(view, scene_to_viewport_point(view, *drop_pt))
            assert scene.placed_items[-1].component.component_type == expected_type
            trigger_placement_hotkey(view, Qt.Key.Key_Escape)
            assert view.placing_component_type is None

        # Pre-rotation during placement via Ctrl+R + Flip via F
        trigger_placement_hotkey(view, Qt.Key.Key_R, scene_pos=(0.0, 100.0))
        press_key(view.viewport(), Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
        click_viewport(view, scene_to_viewport_point(view, 0.0, 100.0))
        r_item = scene.placed_items[-1]
        assert r_item.rotation_deg == 90
        trigger_placement_hotkey(view, Qt.Key.Key_Escape)

        scene.clearSelection()
        r_item.setSelected(True)
        press_key(view.viewport(), Qt.Key.Key_F)
        assert r_item.flipped_h is True

        # Source mode tabs & SPICE preview on DCVoltageSource and ACVoltageSource
        v1 = scene.place_new_component("dc_voltage", QPointF(200.0, 0.0))
        dlg_v = QuickValueDialog(v1.component)
        dlg_v.show()
        qapp.processEvents()
        assert dlg_v.source_mode_tabs is not None
        assert [dlg_v.source_mode_tabs.tabText(i) for i in range(3)] == ["DC", "Sine AC", "Pulse"]
        assert dlg_v.spice_preview_edit is not None and dlg_v.spice_preview_edit.isReadOnly()
        assert "DC" in dlg_v.spice_preview_edit.text()
        dlg_v.source_mode_tabs.setCurrentIndex(1)
        assert "SIN(" in dlg_v.spice_preview_edit.text()
        dlg_v.source_mode_tabs.setCurrentIndex(2)
        assert "PULSE(" in dlg_v.spice_preview_edit.text()

        # Ratings tab edits + engineering warning on 10M / 1mhz
        assert dlg_v.max_power_edit is not None
        dlg_v.max_power_edit.setText("2")
        dlg_v.inputs["dc_voltage"].setText("10M")
        qapp.processEvents()
        assert dlg_v.warning_label is not None and dlg_v.warning_label.isVisible()
        assert "Mega" in dlg_v.warning_label.text()
        dlg_v.close()
    finally:
        scene.undo_stack.setClean()
        win.close()


@pytest.mark.parametrize("zoom", [0.5, 1.0, 2.0])
def test_f1_hotkey_circuit_pin0_polarity_and_dialog_solve(tmp_path, qapp, zoom: float) -> None:
    """F1 end-to-end at zooms 0.5/1.0/2.0: build 12V+500ohm via hotkeys V/R/G/W, rotate 180 deg -> current sign, dialog '2k' -> 6 mA."""
    win, view, scene = create_window_and_view(tmp_path / f"f1_{zoom}", zoom=zoom)
    try:
        # Place V1 (-160, 0) rotated 90 deg via V, Ctrl+R, click, Esc
        trigger_placement_hotkey(view, Qt.Key.Key_V, scene_pos=(-160.0, 0.0))
        press_key(view.viewport(), Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
        click_viewport(view, scene_to_viewport_point(view, -160.0, 0.0))
        trigger_placement_hotkey(view, Qt.Key.Key_Escape)

        # Place R1 (0, -40) via R, click, Esc
        trigger_placement_hotkey(view, Qt.Key.Key_R, scene_pos=(0.0, -40.0))
        click_viewport(view, scene_to_viewport_point(view, 0.0, -40.0))
        trigger_placement_hotkey(view, Qt.Key.Key_Escape)

        # Place GND1 (-160, 100) via G, click, Esc
        trigger_placement_hotkey(view, Qt.Key.Key_G, scene_pos=(-160.0, 100.0))
        click_viewport(view, scene_to_viewport_point(view, -160.0, 100.0))
        trigger_placement_hotkey(view, Qt.Key.Key_Escape)

        assert [it.component.component_type for it in scene.placed_items] == [
            "dc_voltage",
            "resistor",
            "ground",
        ]
        v1, r1, gnd = scene.placed_items

        # Double-click R1 to set 500 Ohm and max_power 0.5 W via dialog
        def _set_500_ohm(dlg: QuickValueDialog) -> None:
            dlg.inputs["resistance"].setText("500")
            assert dlg.max_power_edit is not None
            dlg.max_power_edit.setText("0.5")
            dlg._on_ok_clicked()

        open_component_dialog(view, (r1.pos().x(), r1.pos().y()), dialog_cls=QuickValueDialog, on_dialog=_set_500_ohm)
        assert math.isclose(r1.component.resistance, 500.0, rel_tol=1e-9)
        assert math.isclose(r1.component.max_power, 0.5, rel_tol=1e-9)

        # Wire V1.pos -> R1.a (Pin 0 on left) and R1.b (Pin 1 on right) -> V1.neg, V1.neg -> GND using W hotkey
        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        p_r1a = terminal_scene_point(r1, "a")
        p_r1b = terminal_scene_point(r1, "b")
        p_gnd = terminal_scene_point(gnd, "gnd")

        marks_0 = passive_pin_markers((0.0, -40.0), rotation_deg=0, flipped_h=False)
        assert marks_0["pin0_pos"] == p_r1a
        assert marks_0["pin1_pos"] == p_r1b
        assert marks_0["pin0_tick"][0] < marks_0["pin1_tick"][0]

        trigger_placement_hotkey(view, Qt.Key.Key_W)
        assert view.wire_mode_enabled is True
        draw_wire_click_to_click(view, p_vp, p_r1a)
        draw_wire_click_to_click(view, p_r1b, p_vn)
        draw_wire_click_to_click(view, p_vn, p_gnd)
        trigger_placement_hotkey(view, Qt.Key.Key_Escape)
        assert view.wire_mode_enabled is False

        conn1 = scene.extract_connectivity_result()
        sol1 = MNASolver(conn1.circuit).solve_dc()
        # Positive branch current enters Pin 0 (R1.a) at +12V -> I(R1) = +24.0 mA
        assert math.isclose(sol1.branch_currents["R1"], +0.024, rel_tol=1e-6)

        # Now place an unattached R2, rotate 180 deg, wire geometrically left->right (V1.pos -> geometric left = R1.b!)
        # Actually rotate R1 by 180 deg (Ctrl+R twice), verify TerminalRef swapped left/right & wire R1.b->V1.pos, R1.a->V1.neg
        scene.clearSelection()
        r1.setSelected(True)
        press_key(view.viewport(), Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
        press_key(view.viewport(), Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
        assert r1.rotation_deg == 180
        p_r1a_180 = terminal_scene_point(r1, "a")
        p_r1b_180 = terminal_scene_point(r1, "b")
        marks_180 = passive_pin_markers((0.0, -40.0), rotation_deg=180, flipped_h=False)
        assert marks_180["pin0_pos"] == p_r1a_180 == (40, -40)
        assert marks_180["pin1_pos"] == p_r1b_180 == (-40, -40)
        assert marks_180["pin0_tick"][0] > marks_180["pin1_tick"][0]
        # Rewire R1 backwards (V1.pos to R1.b, V1.neg to R1.a) by deleting existing wires first.
        scene.clearSelection()
        for w in list(scene.placed_wires):
            w.setSelected(True)
        scene.delete_selected(detach_unselected_wires=True)
        trigger_placement_hotkey(view, Qt.Key.Key_W)
        draw_wire_click_to_click(view, p_vp, p_r1b_180)
        draw_wire_click_to_click(view, p_r1a_180, p_vn)
        draw_wire_click_to_click(view, p_vn, p_gnd)
        trigger_placement_hotkey(view, Qt.Key.Key_Escape)

        conn2 = scene.extract_connectivity_result()
        sol2 = MNASolver(conn2.circuit).solve_dc()
        # Geometric left pin is now Pin 1 (b), right is Pin 0 (a) -> current exits Pin 0 -> I(R1) = -24.0 mA!
        assert math.isclose(sol2.branch_currents["R1"], -0.024, rel_tol=1e-6)

        # Double-click R1, set value to "2k" via dialog -> solved current magnitude = 12 / 2000 = 6.000 mA
        def _set_2k(dlg: QuickValueDialog) -> None:
            dlg.inputs["resistance"].setText("2k")
            dlg._on_ok_clicked()

        open_component_dialog(view, (r1.pos().x(), r1.pos().y()), dialog_cls=QuickValueDialog, on_dialog=_set_2k)
        conn3 = scene.extract_connectivity_result()
        frame3 = SimulationSession(connectivity=conn3).start()
        assert math.isclose(abs(frame3.components["R1"].current), 0.006, rel_tol=1e-6)
    finally:
        scene.undo_stack.setClean()
        win.close()

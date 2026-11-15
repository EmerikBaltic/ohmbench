"""Real-input offscreen UI flow reproduction and regression tests for FIX QUEUE (E1-E6)."""

from __future__ import annotations

import math
import sys

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor

from circuitsim.app.themes import DARK_PALETTE, LIGHT_PALETTE, build_stylesheet
from circuitsim.engine import MNASolver
from tests.ui_flow_helpers import (
    click_viewport,
    create_window_and_view,
    drag_viewport,
    draw_wire_click_to_click,
    press_key,
    render_scene_image,
    scene_to_viewport_point,
    select_all_and_delete,
    terminal_scene_point,
)


@pytest.mark.xfail(strict=True, reason="E1 reproduction")
def test_e1_repro_select_all_delete_crash(tmp_path, qapp) -> None:
    """E1 repro: selecting all components/wires/junctions and pressing Delete must not crash."""
    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -40.0))
        vm1 = scene.place_new_component("voltmeter", QPointF(160.0, 0.0), rotation_deg=90)
        a1 = scene.place_new_component("ammeter", QPointF(0.0, 120.0))
        gnd = scene.place_new_component("ground", QPointF(-160.0, 140.0))
        qapp.processEvents()

        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        p_r1a = terminal_scene_point(r1, "a")
        p_r1b = terminal_scene_point(r1, "b")
        p_vm1_pos = terminal_scene_point(vm1, "pos")
        p_vm1_neg = terminal_scene_point(vm1, "neg")
        p_a1a = terminal_scene_point(a1, "pos")
        p_gnd = terminal_scene_point(gnd, "gnd")

        win.act_wire_mode.setChecked(True)
        draw_wire_click_to_click(view, p_vp, p_r1a)
        draw_wire_click_to_click(view, p_r1b, p_vm1_pos)
        draw_wire_click_to_click(view, p_vm1_neg, p_vn)
        draw_wire_click_to_click(view, p_vn, p_gnd)
        # Also add a free-end wire and an extra branch to cover junctions + free ends.
        draw_wire_click_to_click(view, p_a1a, (-40, 200))
        win.act_wire_mode.setChecked(False)
        qapp.processEvents()

        assert len(scene.placed_items) == 5
        assert len(scene.placed_wires) >= 4

        slot_exceptions: list[BaseException] = []
        old_hook = sys.excepthook

        def _capture_hook(exc_type, exc_val, exc_tb) -> None:
            slot_exceptions.append(exc_val)
            old_hook(exc_type, exc_val, exc_tb)

        sys.excepthook = _capture_hook
        try:
            select_all_and_delete(view, use_rubberband=True)
        finally:
            sys.excepthook = old_hook

        assert not slot_exceptions, f"Unhandled slot exception(s) on Delete: {slot_exceptions!r}"
        assert len(scene.placed_items) == 0
        assert len(scene.placed_wires) == 0
    finally:
        scene.undo_stack.setClean()
        win.close()


@pytest.mark.xfail(strict=True, reason="E2 reproduction")
def test_e2_repro_terminal_snap_zoom_and_wire_on_wire(tmp_path, qapp) -> None:
    """E2 repro: zoomed-out terminal clicks and wire-on-wire endings must connect electrically."""
    win, view, scene = create_window_and_view(tmp_path, zoom=0.5)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -40.0))
        gnd = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        r2 = scene.place_new_component("resistor", QPointF(160.0, 40.0), rotation_deg=90)
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 500.0})
        scene.edit_component_parameters(r2, {"resistance": 500.0})
        qapp.processEvents()

        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        p_r1a = terminal_scene_point(r1, "a")
        p_r1b = terminal_scene_point(r1, "b")
        p_r2a = terminal_scene_point(r2, "a")
        p_r2b = terminal_scene_point(r2, "b")
        p_gnd = terminal_scene_point(gnd, "gnd")

        win.act_wire_mode.setChecked(True)
        # Click with +6 view-px offset at zoom=0.5 (12 scene px, inside 9.5 view-px radius).
        draw_wire_click_to_click(view, p_vp, p_r1a, start_offset_vpx=(6, 0), end_offset_vpx=(6, 0))
        # Exact vertical drop on right side: r1.b -> r2.b -> v1.neg
        draw_wire_click_to_click(view, p_r1b, p_vn, waypoints_scene=[(80, -40), (80, 40)])
        draw_wire_click_to_click(view, p_vn, p_gnd)
        # End wire from R2.a onto the interior of the vertical wire at x=80, y=0.
        draw_wire_click_to_click(view, p_r2a, (80, 0))
        draw_wire_click_to_click(view, p_r2b, (80, 40))
        win.act_wire_mode.setChecked(False)
        qapp.processEvents()

        conn = scene.extract_connectivity_result()
        n_vp = conn.component_nodes["V1"]["pos"]
        n_r1a = conn.component_nodes["R1"]["a"]
        n_r2a = conn.component_nodes["R2"]["a"]
        n_vn = conn.component_nodes["V1"]["neg"]
        assert n_r1a == n_vp, f"Expected R1.a connected to V1.pos at zoom=0.5 +6vpx, got {n_r1a} != {n_vp}"
        assert n_r2a == n_vn, f"Expected R2.a tapped onto vertical wire interior to V1.neg, got {n_r2a} != {n_vn}"

        if hasattr(scene, "wire_graph"):
            # On wire-redesign, ensure no spurious FreeEnd at V1.pos / R1.a or wire-on-wire FreeEnd.
            for w in scene.wire_graph.wires:
                assert not (type(w.end_b).__name__ == "FreeEnd" and w.end_b.pos == (80, 0))
    finally:
        scene.undo_stack.setClean()
        win.close()


@pytest.mark.xfail(strict=True, reason="E3 reproduction")
def test_e3_repro_drop_2terminal_part_on_wire(tmp_path, qapp) -> None:
    """E3 repro: dragging a 2-terminal ammeter onto a wire splits inline without shorting it."""
    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        a1 = scene.place_new_component("ammeter", QPointF(0.0, 140.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 500.0})
        qapp.processEvents()

        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        r1_top = min(
            [("a", terminal_scene_point(r1, "a")), ("b", terminal_scene_point(r1, "b"))],
            key=lambda kv: kv[1][1],
        )[1]
        r1_bot = max(
            [("a", terminal_scene_point(r1, "a")), ("b", terminal_scene_point(r1, "b"))],
            key=lambda kv: kv[1][1],
        )[1]
        p_gnd = terminal_scene_point(gnd, "gnd")

        win.act_wire_mode.setChecked(True)
        # Straight horizontal top wire across y = -40: (-160, -40) .. (160, -40).
        draw_wire_click_to_click(view, p_vp, r1_top)
        draw_wire_click_to_click(view, p_vn, r1_bot)
        draw_wire_click_to_click(view, p_vn, p_gnd)
        win.act_wire_mode.setChecked(False)
        qapp.processEvents()

        # Drag ammeter A1 from (0, 140) up across the bottom wire to (0, -40) onto the top wire.
        start_vp = scene_to_viewport_point(view, a1.pos().x(), a1.pos().y())
        drop_vp = scene_to_viewport_point(view, 0.0, -40.0)
        drag_viewport(view, start_vp, drop_vp, steps=12)

        assert (a1.pos().x(), a1.pos().y()) == (0.0, -40.0)
        conn = scene.extract_connectivity_result()
        assert "A1" not in conn.unconnected_components, "Ammeter A1 was left disconnected after drop onto wire!"
        n_a1_a = conn.component_nodes["A1"]["pos"]
        n_a1_b = conn.component_nodes["A1"]["neg"]
        assert n_a1_a != n_a1_b, "Ammeter was shorted by unsplit parallel wire / double junction tap!"

        from circuitsim.engine import SimulationSession

        frame = SimulationSession(connectivity=conn).start()
        i_a1 = abs(frame.components["A1"].current)
        # Hand calc: 12.0 V / 500.0 Ohm = 0.024 A (24.0 mA).
        assert math.isclose(i_a1, 0.024, rel_tol=1e-4), f"Expected 24.0 mA through A1, got {i_a1}"
    finally:
        scene.undo_stack.setClean()
        win.close()


@pytest.mark.xfail(strict=True, reason="E4 reproduction")
def test_e4_repro_wire_mode_visibility_and_esc(tmp_path, qapp) -> None:
    """E4 repro: Wire mode must style :checked buttons, show status text, and exit on Esc."""
    dark_css = build_stylesheet(DARK_PALETTE)
    light_css = build_stylesheet(LIGHT_PALETTE)
    assert "QToolButton:checked" in dark_css
    assert "QToolButton:checked" in light_css

    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(0.0, 0.0))
        qapp.processEvents()
        p_vp = terminal_scene_point(v1, "pos")

        win.act_wire_mode.trigger()
        qapp.processEvents()
        assert view.wire_mode_enabled is True
        assert win.act_wire_mode.isChecked() is True
        status = win.statusBar().currentMessage()
        assert "Wire mode" in status and "Esc" in status, f"Unexpected status bar text: {status!r}"

        # Start a wire on V1.pos, move to empty canvas, press Esc -> preview cancelled AND leaves Wire mode.
        click_viewport(view, scene_to_viewport_point(view, *p_vp))
        assert scene.is_drawing_wire is True
        from PySide6.QtTest import QTest
        QTest.mouseMove(view.viewport(), scene_to_viewport_point(view, -120, -120))
        qapp.processEvents()

        press_key(view.viewport(), Qt.Key.Key_Escape)
        assert scene.is_drawing_wire is False
        assert view.wire_mode_enabled is False
        assert win.act_wire_mode.isChecked() is False
    finally:
        scene.undo_stack.setClean()
        win.close()


@pytest.mark.xfail(strict=True, reason="E5 reproduction")
def test_e5_repro_label_plates_cut_wire(tmp_path, qapp) -> None:
    """E5 repro: opaque label/value plates must not erase a wire passing through the label rect."""
    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        vm = scene.place_new_component("voltmeter", QPointF(0.0, 0.0), rotation_deg=90)
        vm.live_display_value = "12.00 V"
        qapp.processEvents()

        # Rotate 90 puts label/value plates on the right side (x ~ 25..100), straddling 20px grid x=40, 60.
        plate_rect = vm.candidate_label_rects_scene()[1]
        x_grid = 60
        for gx in (40, 60, 80, 20, 100):
            if plate_rect.left() + 3.0 <= gx <= plate_rect.right() - 3.0:
                x_grid = gx
                break

        assert plate_rect.left() + 3.0 <= x_grid <= plate_rect.right() - 3.0, (
            f"No 20px grid line inside value plate {plate_rect}"
        )

        win.act_wire_mode.setChecked(True)
        draw_wire_click_to_click(view, (x_grid, -120), (x_grid, 120))
        win.act_wire_mode.setChecked(False)
        qapp.processEvents()

        render_rect = QRectF(-40.0, -120.0, 200.0, 240.0)
        img = render_scene_image(scene, render_rect, bg_hex=DARK_PALETTE.canvas_bg)

        # Sample a reference wire pixel well outside any component/label plate (x = x_grid, y = -80).
        px_x = int(round(x_grid - render_rect.left()))
        px_ref_y = int(round(-80.0 - render_rect.top()))
        ref_color = img.pixelColor(px_x, px_ref_y)
        canvas_bg = QColor(DARK_PALETTE.canvas_bg)
        assert ref_color != canvas_bg, "Reference wire pixel outside plate was not rendered"

        # Sample inside the top padding of the value plate (inside plate_rect, along the wire).
        inside_y_scene = plate_rect.top() + 0.5
        px_in_y = int(round(inside_y_scene - render_rect.top()))
        inside_color = img.pixelColor(px_x, px_in_y)
        assert inside_color != canvas_bg, (
            f"Label plate overwrote wire at ({x_grid}, {inside_y_scene}) with canvas_bg {canvas_bg.name()}"
        )
    finally:
        scene.undo_stack.setClean()
        win.close()

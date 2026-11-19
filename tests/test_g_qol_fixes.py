"""Tests for Phase G Owner QOL & Bug Fixes (G3-G8)."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import DEFAULT_SHORTCUTS, AppSettings
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.canvas.view import CircuitView
from circuitsim.dialogs.preferences_dialog import PreferencesDialog, ShortcutsDialog
from circuitsim.engine import MNASolver, SimulationSession
from circuitsim.wiring.model import FreeEnd, JunctionRef, TerminalRef
from circuitsim.wiring.router import route as wire_route
from tests.ui_flow_helpers import (
    click_viewport,
    drag_viewport,
    draw_wire_click_to_click,
    press_key,
    scene_to_viewport_point,
    terminal_scene_point,
)


def test_g3_scrollwheel_zoom_anchors_to_viewport_center(qapp) -> None:
    """G3: Scrollwheel zoom in/out stays centered strictly on viewport center without corner drift."""
    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    view.resize(800, 600)
    view.show()
    view.set_view_state(1.0, (100.0, -60.0))
    qapp.processEvents()

    try:
        c0 = view.view_center()
        assert math.isclose(c0[0], 100.0, abs_tol=2.0)
        assert math.isclose(c0[1], -60.0, abs_tol=2.0)

        # Send zoom-in wheel events with cursor near bottom-left corner (where naive AnchorUnderMouse drifts!)
        vp_corner = QPointF(25.0, 575.0)
        for _ in range(4):
            ev_in = QWheelEvent(
                vp_corner,
                view.viewport().mapToGlobal(vp_corner.toPoint()),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(view.viewport(), ev_in)
            qapp.processEvents()
            cx, cy = view.view_center()
            assert math.hypot(cx - 100.0, cy - (-60.0)) <= 2.0

        # Send zoom-out wheel events past initial zoom
        for _ in range(6):
            ev_out = QWheelEvent(
                vp_corner,
                view.viewport().mapToGlobal(vp_corner.toPoint()),
                QPoint(0, 0),
                QPoint(0, -120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(view.viewport(), ev_out)
            qapp.processEvents()
            cx, cy = view.view_center()
            assert math.hypot(cx - 100.0, cy - (-60.0)) <= 2.0
    finally:
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()


def test_g4_hotkey_autorepeat_ignored_and_no_overlapping_placement(qapp) -> None:
    """G4: Auto-repeat hotkeys are ignored and stacked non-voltmeter placements/drops offset cleanly."""
    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    view.resize(900, 650)
    view.show()
    view.set_view_state(1.0, (0.0, 0.0))
    qapp.processEvents()

    try:
        vp_origin = scene_to_viewport_point(view, 0.0, 0.0)
        QTest.mouseMove(view.viewport(), vp_origin)
        qapp.processEvents()

        # Auto-repeat R press when not placing must be ignored (cannot activate placement)
        rep_ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.NoModifier,
            "r",
            True,
            1,
        )
        QApplication.sendEvent(view.viewport(), rep_ev)
        qapp.processEvents()
        assert view.placing_component_type is None

        # Non-repeat R enters placement; subsequent 9 auto-repeat R events stay ignored while 10 clicks stack cleanly
        first_ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.NoModifier,
            "r",
            False,
            1,
        )
        QApplication.sendEvent(view.viewport(), first_ev)
        qapp.processEvents()
        assert view.placing_component_type == "resistor"

        # Repeated clicks on occupied origin are blocked (I1 owner rule: no brick pile!)
        for idx in range(10):
            # Send auto-repeat R presses too to simulate holding the key down
            hold_ev = QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_R,
                Qt.KeyboardModifier.NoModifier,
                "r",
                idx > 0,
                1,
            )
            QApplication.sendEvent(view.viewport(), hold_ev)
            click_viewport(view, vp_origin)
        qapp.processEvents()

        assert len(scene.placed_items) == 1

        # Drag-drop onto an existing non-voltmeter body reverts cleanly back to start position
        r0 = scene.placed_items[0]
        r1 = scene.place_new_component("resistor", QPointF(0.0, 120.0))
        assert r1 is not None
        scene.begin_component_move_tracking()
        r1.setPos(r0.pos())
        assert scene.commit_component_move_tracking() is False
        assert (r1.pos().x(), r1.pos().y()) == (0.0, 120.0)
    finally:
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()


def test_g5_auto_connect_touching_endpoints_and_gnd_tap(qapp) -> None:
    """G5 (resolves E2): Wires touching terminals within 12 px or wire interiors/corners auto-connect without FreeEnds."""
    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    view.resize(1000, 720)
    view.show()
    view.set_view_state(1.0, (0.0, 0.0))
    qapp.processEvents()

    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(-40.0, -40.0))
        d1 = scene.place_new_component("led", QPointF(120.0, -40.0))
        gnd = scene.place_new_component("ground", QPointF(0.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})

        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        p_ra = terminal_scene_point(r1, "a")
        p_rb = terminal_scene_point(r1, "b")
        p_da = terminal_scene_point(d1, "anode")
        p_dk = terminal_scene_point(d1, "cathode")
        p_gnd = terminal_scene_point(gnd, "gnd")

        view.set_wire_mode(True)
        # V1.pos -> R1.a
        draw_wire_click_to_click(view, p_vp, p_ra)
        # R1.b -> within 6 scene px of LED1.anode (tests snap tolerance + auto-connect)
        draw_wire_click_to_click(view, p_rb, (p_da[0] + 6, p_da[1]))
        # Bottom wire: LED1.cathode down and across to V1.neg along y = 40
        draw_wire_click_to_click(view, p_dk, p_vn, waypoints_scene=[(p_dk[0], p_vn[1])])
        # GND wire ends directly on the interior of the bottom wire at (0, 40)
        draw_wire_click_to_click(view, p_gnd, (p_gnd[0], p_vn[1]))
        view.set_wire_mode(False)
        qapp.processEvents()

        # Assert zero FreeEnds remain across the entire circuit graph
        for w in scene.wire_graph.wires:
            assert not isinstance(w.end_a, FreeEnd), f"Unexpected FreeEnd on {w.id}.end_a: {w.end_a}"
            assert not isinstance(w.end_b, FreeEnd), f"Unexpected FreeEnd on {w.id}.end_b: {w.end_b}"

        # Assert a JunctionRef exists at the GND tap (0, 40)
        assert len(scene.wire_graph.junctions) >= 1
        assert any(j.pos == (p_gnd[0], p_vn[1]) for j in scene.wire_graph.junctions)

        conn = scene.extract_connectivity_result()
        assert not conn.unconnected_components
        sol = MNASolver(conn.circuit).solve_dc()
        # 12 V across 1k resistor + ~2.0 V LED -> ~10.0 mA forward current
        assert math.isclose(abs(sol.branch_currents["R1"]), 0.010, abs_tol=1e-3)
    finally:
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()


def test_g6_straight_vertical_and_l_route_without_side_jogs(qapp) -> None:
    """G6: Routing straight down from a horizontal terminal produces a 2-point vertical line without 1-unit side jogs."""
    # Direct Qt-free router check from horizontal terminal (start_dir=(1, 0)) straight down 100 px
    r_straight = wire_route((40, 0), (40, 100), start_dir=(1, 0))
    assert r_straight == ((40, 0), (40, 100))

    # Horizontal-then-vertical L-route reaches x2 cleanly without overshooting x2
    r_l = wire_route((40, 0), (140, 100), start_dir=(1, 0))
    assert r_l == ((40, 0), (140, 0), (140, 100))

    # Through CircuitScene with horizontal resistor R1 and attached vertical wire straight down
    scene = CircuitScene(SymbolLibrary())
    try:
        r1 = scene.place_new_component("resistor", QPointF(0.0, 0.0))
        x_t, y_t = terminal_scene_point(r1, "b")
        w = scene.add_wire(TerminalRef("R1", "b"), FreeEnd(x_t, y_t + 100))
        assert w is not None
        assert w.points == ((x_t, y_t), (x_t, y_t + 100))
        assert all(px == x_t for px, _py in w.points)
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_g7_voltmeter_parallel_overlay_and_no_random_drag_connections(qapp) -> None:
    """G7: Overlaid voltmeter auto-wires in parallel & survives 60px drag; passing parts/wires during drag never mutates WireGraph."""
    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    view.resize(1000, 720)
    view.show()
    view.set_view_state(1.0, (0.0, 0.0))
    qapp.processEvents()

    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(-40.0, -40.0))
        d1 = scene.place_new_component("led", QPointF(120.0, -40.0))
        gnd = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})

        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        p_ra = terminal_scene_point(r1, "a")
        p_rb = terminal_scene_point(r1, "b")
        p_da = terminal_scene_point(d1, "anode")
        p_dk = terminal_scene_point(d1, "cathode")
        p_gnd = terminal_scene_point(gnd, "gnd")

        scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
        scene.add_wire(TerminalRef("R1", "b"), TerminalRef("led1", "anode"))
        scene.add_wire(TerminalRef("led1", "cathode"), TerminalRef("V1", "neg"))
        scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
        qapp.processEvents()

        # (a) Place VM1 overlaid on led1, then drag VM1 60 px straight up and release
        vm1 = scene.place_new_component("voltmeter", d1.pos(), rotation_deg=d1.rotation_deg)
        qapp.processEvents()

        # Drag VM1 straight up by 60 px using real QTest mouse events
        vm_start_vp = scene_to_viewport_point(view, vm1.pos().x(), vm1.pos().y())
        vm_end_vp = scene_to_viewport_point(view, vm1.pos().x(), vm1.pos().y() - 60.0)
        drag_viewport(view, vm_start_vp, vm_end_vp, steps=8)

        assert (vm1.pos().x(), vm1.pos().y()) == (d1.pos().x(), d1.pos().y() - 60.0)
        vm_wires = [
            w
            for w in scene.wire_graph.wires
            if (isinstance(w.end_a, TerminalRef) and w.end_a.component == "VM1")
            or (isinstance(w.end_b, TerminalRef) and w.end_b.component == "VM1")
        ]
        assert len(vm_wires) == 2

        conn = scene.extract_connectivity_result()
        assert conn.component_nodes["VM1"]["pos"] == conn.component_nodes["led1"]["anode"]
        assert conn.component_nodes["VM1"]["neg"] == conn.component_nodes["led1"]["cathode"]
        frame = SimulationSession(connectivity=conn).start()
        assert math.isclose(frame.components["VM1"].voltage, 2.0, abs_tol=0.15)

        # (b) Drag VM2 and A1 directly across R1 and existing wires -> zero WireGraph mutation during move or empty drop
        vm2 = scene.place_new_component("voltmeter", QPointF(-40.0, 120.0))
        a1 = scene.place_new_component("ammeter", QPointF(80.0, 120.0))
        qapp.processEvents()
        graph_before = scene.wire_graph

        for comp_item, target_empty_y in ((vm2, -140.0), (a1, -140.0)):
            cx = comp_item.pos().x()
            p0 = scene_to_viewport_point(view, cx, comp_item.pos().y())
            p_mid = scene_to_viewport_point(view, cx, -40.0)  # right over R1 / wires
            p_end = scene_to_viewport_point(view, cx, target_empty_y)
            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
            qapp.processEvents()
            assert scene.wire_graph == graph_before
            QTest.mouseMove(view.viewport(), p_mid)
            qapp.processEvents()
            assert scene.wire_graph == graph_before
            QTest.mouseMove(view.viewport(), p_end)
            qapp.processEvents()
            assert scene.wire_graph == graph_before
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p_end)
            qapp.processEvents()
            assert scene.wire_graph == graph_before
    finally:
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()


def test_g8_programmable_shortcuts_tab_duplicate_resolution_and_dispatch(tmp_path, qapp) -> None:
    """G8: AppSettings.shortcuts, Preferences/ShortcutsDialog rebinding, duplicate resolution, and reset to defaults."""
    data_dir = tmp_path / "g8_settings"
    settings = AppSettings(data_dir=data_dir)
    for required_key in (
        "rotate",
        "flip",
        "wire_mode",
        "junction",
        "delete",
        "place_resistor",
        "place_capacitor",
        "place_inductor",
        "place_diode",
        "place_dc_voltage",
        "place_ground",
        "place_voltmeter",
        "place_ammeter",
    ):
        assert required_key in settings.shortcuts

    win = MainWindow(settings=settings)
    win.resize(1000, 720)
    win.show()
    qapp.processEvents()
    view = win.circuit_view
    scene = win.circuit_scene
    view.set_view_state(1.0, (0.0, 0.0))

    try:
        # PreferencesDialog includes General + Shortcuts tabs
        pref_dlg = PreferencesDialog(settings, win)
        assert pref_dlg.tabs.count() >= 2
        assert pref_dlg.tabs.tabText(1) == "Shortcuts"
        pref_dlg.close()

        # Test duplicate conflict resolution inside ShortcutsDialog
        sc_dlg = ShortcutsDialog(settings, win)
        sc_dlg.editor.set_binding("rotate", "F")  # conflicts with default flip='F'
        assert sc_dlg.editor.get_bindings()["flip"] == ""  # auto-cleared duplicate
        assert "Reassigned" in sc_dlg.editor.conflict_label.text()

        # Rebind rotate -> 'Q', flip -> 'X', place_resistor -> 'T' and save
        sc_dlg.editor.set_binding("rotate", "Q")
        sc_dlg.editor.set_binding("flip", "X")
        sc_dlg.editor.set_binding("place_resistor", "T")
        sc_dlg._on_accept()
        sc_dlg.close()
        qapp.processEvents()

        assert settings.shortcuts["rotate"] == "Q"
        assert settings.shortcuts["flip"] == "X"
        assert settings.shortcuts["place_resistor"] == "T"
        # Persistence check via fresh AppSettings instance
        assert AppSettings(data_dir=data_dir).shortcuts["rotate"] == "Q"

        # Old 'R' no longer enters resistor placement; new 'T' enters resistor placement
        press_key(view.viewport(), Qt.Key.Key_R)
        assert view.placing_component_type is None

        press_key(view.viewport(), Qt.Key.Key_T)
        assert view.placing_component_type == "resistor"
        click_viewport(view, scene_to_viewport_point(view, 0.0, 0.0))
        press_key(view.viewport(), Qt.Key.Key_Escape)
        assert len(scene.placed_items) == 1
        r1 = scene.placed_items[0]

        # Trigger rebound rotate ('Q') and flip ('X') in CircuitView
        scene.clearSelection()
        r1.setSelected(True)
        press_key(view.viewport(), Qt.Key.Key_Q)
        assert r1.rotation_deg == 90
        press_key(view.viewport(), Qt.Key.Key_X)
        assert r1.flipped_h is True

        # Reset to Defaults via UI dialog button
        sc_dlg2 = ShortcutsDialog(settings, win)
        sc_dlg2.editor.btn_reset_defaults.click()
        sc_dlg2._on_accept()
        sc_dlg2.close()
        qapp.processEvents()
        assert settings.shortcuts == DEFAULT_SHORTCUTS

        # Default 'R' works again
        press_key(view.viewport(), Qt.Key.Key_R)
        assert view.placing_component_type == "resistor"
        press_key(view.viewport(), Qt.Key.Key_Escape)
    finally:
        win._is_closing = True
        win.close()
        qapp.processEvents()

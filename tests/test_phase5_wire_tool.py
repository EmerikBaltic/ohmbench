"""Tests for Phase 5 (`Step 4A-2`): interactive wire drawing, waypoints, segment drag, and end editing."""

from __future__ import annotations

import math
import os
import statistics
import time

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.engine import MNASolver, SimulationSession, SimulationState
from circuitsim.wiring.model import FreeEnd, Junction, JunctionRef, TerminalRef, WireGraph
from circuitsim.wiring.router import route as wire_route, slide_segment_hints


def _vpos(win: MainWindow, sx: float, sy: float):
    return win.circuit_view.mapFromScene(QPointF(sx, sy))


def test_phase5_1_interactive_wire_drawing_preview_snap_junction_free_end_and_cancel(
    tmp_path, qapp
) -> None:
    """Phase 5.1: interactive wire drawing uses route(), 10px snap, W-mode sync, Esc/RMB cancel, FreeEnd + wire-on-wire notice."""
    settings = AppSettings(data_dir=tmp_path / "settings_51")
    win = MainWindow(settings=settings)
    win.resize(1000, 760)
    win.show()
    qapp.processEvents()
    view = win.circuit_view
    scene = win.circuit_scene
    vp = view.viewport()

    try:
        view.reset_zoom()
        qapp.processEvents()

        # 1. W key and toolbar act_wire_mode stay synchronized; Esc exits W mode.
        assert not view.wire_mode_enabled
        assert not win.act_wire_mode.isChecked()

        QTest.keyClick(view, Qt.Key.Key_W)
        assert view.wire_mode_enabled is True
        assert win.act_wire_mode.isChecked() is True

        QTest.keyClick(view, Qt.Key.Key_Escape)
        assert view.wire_mode_enabled is False
        assert win.act_wire_mode.isChecked() is False

        win.act_wire_mode.setChecked(True)
        assert view.wire_mode_enabled is True
        win.act_wire_mode.setChecked(False)
        assert view.wire_mode_enabled is False

        # 2. Place components for 15 V / 1 kOhm / 2 kOhm divider -> V_mid = 15 * 2k / (1k + 2k) = 10.0 V.
        v1 = scene.place_new_component("dc_voltage", QPointF(-120.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -40.0))
        r2 = scene.place_new_component("resistor", QPointF(120.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-120.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 15.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})

        p_vpos = v1.terminal_scene_positions_px()["pos"]
        p_vneg = v1.terminal_scene_positions_px()["neg"]
        p_r1a = r1.terminal_scene_positions_px()["a"]
        p_r1b = r1.terminal_scene_positions_px()["b"]
        p_r2a = r2.terminal_scene_positions_px()["a"]
        p_r2b = r2.terminal_scene_positions_px()["b"]
        p_gnd = gnd.terminal_scene_positions_px()["gnd"]

        # 3. Dragging directly from terminal (without W mode) shows live routed preview & commits in 1 undo step.
        undo_0 = scene.undo_stack.count()
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, p_vpos[0], p_vpos[1]),
        )
        assert scene.is_drawing_wire is True

        # Drag within 7 px of R1.a -> preview snaps to R1.a and matches final route() points.
        target_near_r1a = (p_r1a[0] - 5, p_r1a[1] + 5)  # hypot(5, 5) ~= 7.07 <= 10 px
        QTest.mouseMove(vp, _vpos(win, *target_near_r1a))
        preview_item = scene._preview_wire_item
        assert preview_item is not None
        preview_pts = preview_item.points
        assert preview_pts[0] == p_vpos and preview_pts[-1] == p_r1a
        assert scene._hovered_terminal_pt == p_r1a

        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, *target_near_r1a),
        )
        assert scene.is_drawing_wire is False
        assert scene.undo_stack.count() == undo_0 + 1
        assert len(scene.placed_wires) == 1
        w_vr1 = scene.placed_wires[0]
        assert w_vr1.end_a == TerminalRef("V1", "pos")
        assert w_vr1.end_b == TerminalRef("R1", "a")
        assert w_vr1.points == preview_pts

        # 4. Esc and right-click cancel wire preview cleanly with 0 undo steps and no context menu popup.
        QTest.keyClick(view, Qt.Key.Key_W)
        assert view.wire_mode_enabled is True
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, p_r1b[0], p_r1b[1]),
        )
        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, p_r1b[0], p_r1b[1]),
        )
        assert scene.is_drawing_wire is True
        QTest.mouseMove(vp, _vpos(win, p_r2a[0], p_r2a[1]))
        assert scene._preview_wire_item is not None
        QTest.keyClick(view, Qt.Key.Key_Escape)
        assert scene.is_drawing_wire is False
        assert scene._preview_wire_item is None
        assert scene.undo_stack.count() == undo_0 + 1
        # Esc while preview was active cancels preview and exits W mode cleanly (E4).
        assert view.wire_mode_enabled is False
        QTest.keyClick(view, Qt.Key.Key_W)
        assert view.wire_mode_enabled is True

        # Start again and cancel with RightButton -> no context menu opens & 0 undo steps.
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, p_r1b[0], p_r1b[1]),
        )
        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, p_r1b[0], p_r1b[1]),
        )
        assert scene.is_drawing_wire is True
        QTest.mousePress(
            vp,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, p_r2a[0], p_r2a[1]),
        )
        assert scene.is_drawing_wire is False
        assert view._suppress_next_context_menu is True
        ctx_ev = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            _vpos(win, p_r2a[0], p_r2a[1]),
            vp.mapToGlobal(_vpos(win, p_r2a[0], p_r2a[1])),
        )
        qapp.sendEvent(view.viewport(), ctx_ev)
        assert view._suppress_next_context_menu is False
        assert scene.undo_stack.count() == undo_0 + 1

        # 5. Complete divider wires via click-to-click and drag in W mode; simulate -> 10.0 V.
        w_r1r2 = scene.finish_wire_preview(p_r2a) if scene.is_drawing_wire else None
        assert w_r1r2 is None  # was cancelled
        scene.start_wire_preview(p_r1b)
        scene.finish_wire_preview(p_r2a)
        scene.start_wire_preview(p_r2b)
        scene.finish_wire_preview(p_vneg)
        scene.start_wire_preview(p_vneg)
        scene.finish_wire_preview(p_gnd)

        conn = scene.extract_connectivity_result()
        sol = MNASolver(conn.circuit).solve_dc()
        # Hand calculation: V_mid = 15.0 * 2000.0 / (1000.0 + 2000.0) = 10.0 V
        assert math.isclose(sol.node_voltages[conn.component_nodes["R2"]["a"]], 10.0, rel_tol=1e-6)
        sess = SimulationSession(connectivity=conn)
        frame = sess.start()
        assert frame.state == SimulationState.RUNNING
        assert math.isclose(abs(frame.components["R2"].voltage), 10.0, rel_tol=1e-6)

        # 6. Nearest-terminal-wins when two terminals are close, and FreeEnd when outside 10 px.
        ra = scene.place_new_component("resistor", QPointF(200.0, 200.0))
        rb = scene.place_new_component("resistor", QPointF(300.0, 200.0))
        t_ra_b = ra.terminal_scene_positions_px()["b"]  # (240, 200)
        t_rb_a = rb.terminal_scene_positions_px()["a"]  # (260, 200)

        scene.start_wire_preview((200, 300))
        scene.update_wire_preview((246, 200))  # dist to t_ra_b = 6 px, dist to t_rb_a = 14 px
        assert scene._hovered_terminal_pt == t_ra_b
        w_near_a = scene.finish_wire_preview((246, 200))
        assert w_near_a is not None
        assert w_near_a.end_b == TerminalRef(ra.component.name, "b")

        # Point equidistantly biased toward rb.a (at x=254: 6 px from rb.a, 14 px from ra.b)
        scene.start_wire_preview((220, 300))
        scene.update_wire_preview((254, 200))
        assert scene._hovered_terminal_pt == t_rb_a
        w_near_b = scene.finish_wire_preview((254, 200))
        assert w_near_b is not None
        assert w_near_b.end_b == TerminalRef(rb.component.name, "a")

        # Point > 10 px from terminals in empty space creates FreeEnd(gx, gy).
        undo_before_free = scene.undo_stack.count()
        scene.start_wire_preview((300, 300))
        w_free = scene.finish_wire_preview((322, 338))
        assert w_free is not None
        assert scene.undo_stack.count() == undo_before_free + 1
        assert w_free.end_a == FreeEnd(300, 300)
        assert w_free.end_b == FreeEnd(320, 340)

        # 7. Landing on an existing Junction within 10 px attaches JunctionRef (and starting in W mode too).
        split_res = scene.wire_graph.split_wire_at(w_vr1.wire_id, (-80, -40))
        scene.set_wire_graph(split_res.graph)
        jid = split_res.junction_id

        scene.start_wire_preview((360, 360))
        w_to_j = scene.finish_wire_preview((-75, -36))  # hypot(5, 4) ~= 6.4 <= 10 px
        assert w_to_j is not None
        assert w_to_j.end_b == JunctionRef(jid)

        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, -76, -42),  # within 10 px of junction in W mode
        )
        QTest.mouseMove(vp, _vpos(win, -80, 180))
        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, -80, 180),
        )
        assert scene.is_drawing_wire is False
        w_from_j = scene.placed_wires[-1]
        assert w_from_j.end_a == JunctionRef(jid)
        assert w_from_j.end_b == FreeEnd(-80, 180)

        # 8. Ending on the interior of another wire auto-splits into a junction (updated in G5).
        w_target = scene.add_wire(FreeEnd(0, 260), FreeEnd(160, 260))
        assert w_target is not None
        wires_before = len(scene.wire_graph.wires)
        junctions_before = len(scene.wire_graph.junctions)

        scene.start_wire_preview((80, 320))
        w_on_wire = scene.finish_wire_preview((80, 260))
        assert w_on_wire is not None
        assert isinstance(w_on_wire.end_b, JunctionRef)
        assert len(scene.wire_graph.wires) == wires_before + 2
        assert len(scene.wire_graph.junctions) == junctions_before + 1

        # Undo reverts w_on_wire and the junction split in 1 undo step.
        scene.undo_stack.undo()
        assert len(scene.wire_graph.wires) == wires_before
        assert len(scene.wire_graph.junctions) == junctions_before
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()


def test_phase5_2_waypoints_backspace_and_invalid_wire_rejections(tmp_path, qapp) -> None:
    """Phase 5.2: waypoint clicks, Backspace pop/cancel, double-click finish, and readable zero-undo rejections."""
    settings = AppSettings(data_dir=tmp_path / "settings_52")
    win = MainWindow(settings=settings)
    win.resize(1000, 760)
    win.show()
    qapp.processEvents()
    view = win.circuit_view
    scene = win.circuit_scene
    vp = view.viewport()

    try:
        view.reset_zoom()
        qapp.processEvents()

        r1 = scene.place_new_component("resistor", QPointF(-100.0, 0.0))
        r2 = scene.place_new_component("resistor", QPointF(100.0, 0.0))
        p_r1a = r1.terminal_scene_positions_px()["a"]
        p_r1b = r1.terminal_scene_positions_px()["b"]
        p_r2a = r2.terminal_scene_positions_px()["a"]
        p_r2b = r2.terminal_scene_positions_px()["b"]

        # 1. Click-to-click with intermediate waypoints and Backspace pop (without deleting selected component).
        r2.setSelected(True)
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r1b)
        )
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r1b)
        )
        assert scene.is_drawing_wire is True

        # Add two waypoints (-20, -80) and (40, -80), then pop the second with Backspace.
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, -20, -80)
        )
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, -20, -80)
        )
        assert scene._wire_hints == [(-20, -80)]

        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 40, -80)
        )
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 40, -80)
        )
        assert scene._wire_hints == [(-20, -80), (40, -80)]

        QTest.keyClick(view, Qt.Key.Key_Backspace)
        assert scene.is_drawing_wire is True
        assert scene._wire_hints == [(-20, -80)]
        assert len(scene.placed_items) == 2  # selected R2 was NOT deleted!

        # Add (60, -80) and finish by clicking R2.a
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 60, -80)
        )
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 60, -80)
        )
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r2a)
        )
        assert scene.is_drawing_wire is False
        assert len(scene.placed_wires) == 1
        w_wp = scene.placed_wires[0]
        assert w_wp.end_a == TerminalRef("R1", "b")
        assert w_wp.end_b == TerminalRef("R2", "a")
        assert w_wp.hints == ((-20, -80), (60, -80))
        assert (-20, -80) in w_wp.points and (60, -80) in w_wp.points

        # Backspace with no waypoints cancels active wire preview without deleting selection.
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r1a)
        )
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r1a)
        )
        assert scene.is_drawing_wire is True
        QTest.keyClick(view, Qt.Key.Key_Backspace)
        assert scene.is_drawing_wire is False
        assert len(scene.placed_items) == 2
        r2.setSelected(False)

        # 2. W mode empty-grid start + waypoint + double-click finish -> clean hints & FreeEnd.
        QTest.keyClick(view, Qt.Key.Key_W)
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 0, 120)
        )
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 0, 120)
        )
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 80, 120)
        )
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 80, 120)
        )
        QTest.mouseDClick(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 80, 180)
        )
        assert scene.is_drawing_wire is False
        w_dclick = scene.placed_wires[-1]
        assert w_dclick.end_a == FreeEnd(0, 120)
        assert w_dclick.end_b == FreeEnd(80, 180)
        assert w_dclick.hints == ((80, 120),)

        # 3. Rejections emit readable notices and push ZERO undo steps:
        undo_before = scene.undo_stack.count()
        wires_before = len(scene.placed_wires)

        # 3a. Same terminal
        scene.start_wire_preview(p_r1a)
        assert scene.finish_wire_preview(p_r1a) is None
        assert "terminal to itself" in scene.last_status_notice.lower()
        assert scene.undo_stack.count() == undo_before
        assert len(scene.placed_wires) == wires_before

        # 3b. Same junction
        split_res = scene.wire_graph.split_wire_at(w_wp.wire_id, (20, -80))
        scene.set_wire_graph(split_res.graph)
        wires_after_split = len(scene.placed_wires)
        scene.start_wire_preview((20, -80))
        assert scene.finish_wire_preview((24, -82)) is None
        assert "junction to itself" in scene.last_status_notice.lower()
        assert scene.undo_stack.count() == undo_before
        assert len(scene.placed_wires) == wires_after_split

        # 3c. Duplicate wire between existing attachment pair (R1.a <-> R2.b added once, rejected on 2nd)
        w_dup_first = scene.add_wire(TerminalRef("R1", "a"), TerminalRef("R2", "b"))
        assert w_dup_first is not None
        undo_after_first = scene.undo_stack.count()
        wires_after_first = len(scene.placed_wires)

        scene.start_wire_preview(p_r2b)
        assert scene.finish_wire_preview(p_r1a) is None
        assert "already exists" in scene.last_status_notice.lower()
        assert win.statusBar().currentMessage() == scene.last_status_notice
        assert scene.undo_stack.count() == undo_after_first
        assert len(scene.placed_wires) == wires_after_first

        # 3d. Zero-length wire in empty space
        scene.start_wire_preview((220, 220))
        assert scene.finish_wire_preview((223, 218)) is None
        assert "zero-length" in scene.last_status_notice.lower()
        assert scene.undo_stack.count() == undo_after_first
        assert len(scene.placed_wires) == wires_after_first
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()


def test_phase5_3_wire_segment_dragging_updates_hints_and_undo_redo(
    tmp_path, qapp
) -> None:
    """Phase 5.3: dragging a horizontal or vertical wire segment slides it via hints + route() with 1 undo step."""
    # Unit checks for Qt-free slide_segment_hints helper (grid pitch = 20 px):
    # 1. Z-shape interior V segment at x=60 dragged to x=87 -> snaps to round(87/20)*20 = 80.
    z_pts = ((0, 0), (60, 0), (60, 100), (120, 100))
    assert slide_segment_hints(z_pts, 1, "V", 87) == ((80, 0), (80, 100))

    # 2. L-shape H segment at y=0 dragged to y=-40 -> hints ((0, -40), (100, -40)).
    l_pts = ((0, 0), (100, 0), (100, 100))
    assert slide_segment_hints(l_pts, 0, "H", -40) == ((0, -40), (100, -40))

    # 3. Straight 2-point H segment at y=0 dragged to y=60, then back to y=0 -> hints cleared.
    s_pts = ((0, 0), (100, 0))
    assert slide_segment_hints(s_pts, 0, "H", 60) == ((0, 60), (100, 60))
    assert slide_segment_hints(s_pts, 0, "H", 0) == ()

    settings = AppSettings(data_dir=tmp_path / "settings_53")
    win = MainWindow(settings=settings)
    win.resize(1000, 760)
    win.show()
    qapp.processEvents()
    view = win.circuit_view
    scene = win.circuit_scene
    vp = view.viewport()

    try:
        view.reset_zoom()
        qapp.processEvents()

        # Build 10 V / 1 kOhm / 2 kOhm divider -> V_mid = 10 * 2k / (1k + 2k) = 20/3 V.
        v1 = scene.place_new_component("dc_voltage", QPointF(-140.0, 20.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -60.0))
        r2 = scene.place_new_component("resistor", QPointF(160.0, 20.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-140.0, 140.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})

        p_r1b = r1.terminal_scene_positions_px()["b"]
        p_r2a = r2.terminal_scene_positions_px()["a"]
        p_vpos = v1.terminal_scene_positions_px()["pos"]

        # Create a 3-segment H-V-H wire from R1.b to R2.a using a middle vertical hint x_mid.
        x_mid = ((p_r1b[0] + p_r2a[0]) // 40) * 20
        w_top = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
        w_mid = scene.add_wire(
            TerminalRef("R1", "b"),
            TerminalRef("R2", "a"),
            hints=((x_mid, p_r1b[1]), (x_mid, p_r2a[1])),
        )
        w_bot = scene.add_wire(TerminalRef("R2", "b"), TerminalRef("V1", "neg"))
        w_gnd = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
        assert w_top is not None and w_mid is not None and w_bot is not None and w_gnd is not None

        # 1. Hover & drag interior vertical segment of w_mid horizontally (+40 px -> x_mid + 40).
        assert len(w_mid.points) == 4
        v_seg_mid_y = (w_mid.points[1][1] + w_mid.points[2][1]) / 2.0
        QTest.mouseMove(vp, _vpos(win, x_mid, v_seg_mid_y))
        assert view.cursor().shape() == Qt.CursorShape.SizeHorCursor

        undo_before_v = scene.undo_stack.count()
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, x_mid, v_seg_mid_y),
        )
        assert scene.is_dragging_wire_segment is True
        assert scene.undo_stack.count() == undo_before_v

        new_x = x_mid + 40
        QTest.mouseMove(vp, _vpos(win, new_x, v_seg_mid_y))
        live_pts = scene._wire_items_by_id[w_mid.wire_id].points
        assert (new_x, p_r1b[1]) in live_pts and (new_x, p_r2a[1]) in live_pts
        assert scene.undo_stack.count() == undo_before_v

        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, new_x, v_seg_mid_y),
        )
        assert scene.is_dragging_wire_segment is False
        assert scene.undo_stack.count() == undo_before_v + 1
        assert scene.wire_graph.wire(w_mid.wire_id).hints == (
            (new_x, p_r1b[1]),
            (new_x, p_r2a[1]),
        )

        # 2. Undo and redo restore exact hints and points in 1 step each.
        scene.undo_stack.undo()
        assert scene.wire_graph.wire(w_mid.wire_id).hints == (
            (x_mid, p_r1b[1]),
            (x_mid, p_r2a[1]),
        )
        scene.undo_stack.redo()
        assert scene.wire_graph.wire(w_mid.wire_id).hints == (
            (new_x, p_r1b[1]),
            (new_x, p_r2a[1]),
        )

        # 3. Hover & drag horizontal bottom wire segment vertically (-40 px -> bottom_y - 40).
        w_bot_item = scene._wire_items_by_id[w_bot.wire_id]
        h_idx = next(
            i
            for i in range(len(w_bot_item.points) - 1)
            if w_bot_item.points[i][1] == w_bot_item.points[i + 1][1]
            and abs(w_bot_item.points[i][0] - w_bot_item.points[i + 1][0]) >= 40
        )
        hp1, hp2 = w_bot_item.points[h_idx], w_bot_item.points[h_idx + 1]
        hx = (hp1[0] + hp2[0]) / 2.0
        hy = float(hp1[1])
        QTest.mouseMove(vp, _vpos(win, hx, hy))
        assert view.cursor().shape() == Qt.CursorShape.SizeVerCursor

        undo_before_h = scene.undo_stack.count()
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, hx, hy),
        )
        assert scene.is_dragging_wire_segment is True
        target_hy = int(hy) + 40
        QTest.mouseMove(vp, _vpos(win, hx, target_hy))
        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, hx, target_hy),
        )
        assert scene.is_dragging_wire_segment is False
        assert scene.undo_stack.count() == undo_before_h + 1
        bot_hints = scene.wire_graph.wire(w_bot.wire_id).hints
        assert any(pt[1] == target_hy for pt in bot_hints)

        # 4. Tiny sub-half-grid drag that lands back on the original coordinate pushes 0 undo steps.
        undo_before_noop = scene.undo_stack.count()
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, new_x, v_seg_mid_y),
        )
        QTest.mouseMove(vp, _vpos(win, new_x + 3, v_seg_mid_y))
        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, new_x + 3, v_seg_mid_y),
        )
        assert scene.undo_stack.count() == undo_before_noop

        # 5. Esc cancel mid-drag restores original hints & pushes 0 undo steps.
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, new_x, v_seg_mid_y),
        )
        QTest.mouseMove(vp, _vpos(win, new_x + 60, v_seg_mid_y))
        assert scene.is_dragging_wire_segment is True
        QTest.keyClick(view, Qt.Key.Key_Escape)
        assert scene.is_dragging_wire_segment is False
        assert scene.undo_stack.count() == undo_before_noop
        assert scene.wire_graph.wire(w_mid.wire_id).hints == (
            (new_x, p_r1b[1]),
            (new_x, p_r2a[1]),
        )

        # 6. Right-click cancel mid-drag also restores geometry, pushes 0 undo, suppresses context menu.
        QTest.mousePress(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, new_x, v_seg_mid_y),
        )
        QTest.mouseMove(vp, _vpos(win, new_x + 60, v_seg_mid_y))
        QTest.mousePress(
            vp,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, new_x + 60, v_seg_mid_y),
        )
        assert scene.is_dragging_wire_segment is False
        assert scene.undo_stack.count() == undo_before_noop
        ctx = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            _vpos(win, new_x + 60, v_seg_mid_y),
            vp.mapToGlobal(_vpos(win, new_x + 60, v_seg_mid_y)),
        )
        view.contextMenuEvent(ctx)
        assert ctx.isAccepted() is True

        # 7. Clicking right near a terminal (<= 10 px) starts wire drawing, not segment drag.
        assert scene.find_wire_segment_near((p_vpos[0] + 4, p_vpos[1])) is None

        # 8. Electrical connectivity remains intact after sliding segments: V_mid = 10 * 2k / (1k + 2k) = 20 / 3 V.
        conn = scene.extract_connectivity_result()
        sol = MNASolver(conn.circuit).solve_dc()
        assert math.isclose(
            sol.node_voltages[conn.component_nodes["R1"]["b"]], 20.0 / 3.0, rel_tol=1e-6
        )
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()


def test_phase5_4_wire_endpoint_editing_acceptance_s3_and_snap_orientations(
    tmp_path, qapp
) -> None:
    """Phase 5.4: Acceptance S3 endpoint drag/detach/reattach, 8-orientation 10px snap, junction merge, rejections & cancel."""
    from circuitsim.wiring.model import WireGraphError

    # 1. Unit tests for WireGraph.reconnect_wire_end (including degree-3 -> 2 junction merge).
    g = WireGraph()
    g = g.add_junction((100, 100), junction_id="J1")
    g = g.add_wire(TerminalRef("R1", "a"), JunctionRef("J1"), wire_id="W1")
    g = g.add_wire(JunctionRef("J1"), TerminalRef("R2", "a"), wire_id="W2")
    g = g.add_wire(JunctionRef("J1"), TerminalRef("R3", "a"), wire_id="W3")
    assert g.junction_degree("J1") == 3

    # Reconnect W3 from J1 to FreeEnd(200, 200) -> J1 drops to degree 2 and merges W1+W2 cleanly.
    g2 = g.reconnect_wire_end("W3", "end_a", FreeEnd(200, 200))
    assert not g2.has_junction("J1")
    assert g2.has_wire("W1") and not g2.has_wire("W2") and g2.has_wire("W3")
    assert g2.wire("W1").end_a == TerminalRef("R1", "a")
    assert g2.wire("W1").end_b == TerminalRef("R2", "a")
    assert g2.wire("W1").hints == ((100, 100),)
    assert g2.wire("W3").end_a == FreeEnd(200, 200)

    # Reject same-end and duplicate reconnection.
    try:
        g2.reconnect_wire_end("W1", "end_b", TerminalRef("R1", "a"))
        raise AssertionError("Expected WireGraphError on same-end reconnect")
    except WireGraphError:
        pass
    g_dup = g2.add_wire(TerminalRef("R1", "a"), TerminalRef("R3", "a"), wire_id="W4")
    try:
        g_dup.reconnect_wire_end("W1", "end_b", TerminalRef("R3", "a"))
        raise AssertionError("Expected WireGraphError on duplicate reconnect")
    except WireGraphError:
        pass

    # 2. Synthetic QTest Acceptance S3 in MainWindow:
    # Divider: 12 V source + R1=1kOhm + R2=1kOhm -> V_mid = 12 * 1k / (1k + 1k) = 6.0 V.
    # Reattach to R3=3kOhm -> V_mid = 12 * 3k / (1k + 3k) = 9.0 V.
    settings = AppSettings(data_dir=tmp_path / "settings_54")
    win = MainWindow(settings=settings)
    win.resize(1200, 860)
    win.show()
    qapp.processEvents()
    view = win.circuit_view
    scene = win.circuit_scene
    vp = view.viewport()

    try:
        view.reset_zoom()
        qapp.processEvents()
        assert not view.wire_mode_enabled

        v1 = scene.place_new_component("dc_voltage", QPointF(-200.0, 20.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(-40.0, -60.0))
        r2a = scene.place_new_component("resistor", QPointF(80.0, 40.0), rotation_deg=90)
        r2b = scene.place_new_component("resistor", QPointF(160.0, 40.0), rotation_deg=90)
        r3a = scene.place_new_component("resistor", QPointF(260.0, 40.0), rotation_deg=90)
        r3b = scene.place_new_component("resistor", QPointF(340.0, 40.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-200.0, 140.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        # R2a || R2b = 2k || 2k = 1 kOhm -> V_mid = 12 * 1k / (1k + 1k) = 6.0 V
        scene.edit_component_parameters(r2a, {"resistance": 2000.0})
        scene.edit_component_parameters(r2b, {"resistance": 2000.0})
        # R3a || R3b = 6k || 6k = 3 kOhm -> V_mid = 12 * 3k / (1k + 3k) = 9.0 V
        scene.edit_component_parameters(r3a, {"resistance": 6000.0})
        scene.edit_component_parameters(r3b, {"resistance": 6000.0})

        w_top = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
        w_mid = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("R2", "a"))
        w_2top = scene.add_wire(TerminalRef("R2", "a"), TerminalRef("R3", "a"))
        w_2a_g = scene.add_wire(TerminalRef("R2", "b"), TerminalRef("V1", "neg"))
        w_2b_g = scene.add_wire(TerminalRef("R3", "b"), TerminalRef("V1", "neg"))
        w_3top = scene.add_wire(TerminalRef("R4", "a"), TerminalRef("R5", "a"))
        w_3a_g = scene.add_wire(TerminalRef("R4", "b"), TerminalRef("V1", "neg"))
        w_3b_g = scene.add_wire(TerminalRef("R5", "b"), TerminalRef("V1", "neg"))
        w_gnd = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
        assert all(
            w is not None
            for w in (w_top, w_mid, w_2top, w_2a_g, w_2b_g, w_3top, w_3a_g, w_3b_g, w_gnd)
        )
        assert w_mid is not None

        conn_init = scene.extract_connectivity_result()
        sol_init = MNASolver(conn_init.circuit).solve_dc()
        assert math.isclose(
            sol_init.node_voltages[conn_init.component_nodes["R1"]["b"]], 6.0, rel_tol=1e-6
        )

        p_r1b = r1.terminal_scene_positions_px()["b"]
        p_r2a = r2a.terminal_scene_positions_px()["a"]
        p_r3a = r3a.terminal_scene_positions_px()["a"]

        # Select w_mid so its endpoint handle at R2.a wins unambiguously over w_2top.
        scene._wire_items_by_id[w_mid.wire_id].setSelected(True)

        # Drag w_mid end off R2.a into empty space (120, -120) -> FreeEnd in 1 undo step.
        QTest.mouseMove(vp, _vpos(win, *p_r2a))
        assert scene.hovered_wire_end_pt == p_r2a
        assert view.cursor().shape() == Qt.CursorShape.CrossCursor

        undo_s3 = scene.undo_stack.count()
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r2a)
        )
        assert scene.is_dragging_wire_endpoint is True
        QTest.mouseMove(vp, _vpos(win, 120, -120))
        live_w_mid = scene._wire_items_by_id[w_mid.wire_id]
        assert live_w_mid.end_b == FreeEnd(120, -120)
        assert scene.undo_stack.count() == undo_s3

        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 120, -120)
        )
        assert scene.is_dragging_wire_endpoint is False
        assert scene.undo_stack.count() == undo_s3 + 1
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == FreeEnd(120, -120)

        # Open wire: R1.b is disconnected from R2.a & R4.a and w_mid appears in open_wire_ids.
        conn_open = scene.extract_connectivity_result()
        assert (
            conn_open.component_nodes["R1"]["b"] != conn_open.component_nodes["R2"]["a"]
        )
        assert w_mid.wire_id in conn_open.open_wire_ids

        # Drag FreeEnd(120, -120) onto R4.a (offset +5, -4 => dist ~= 6.4 <= 10 px).
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 120, -120)
        )
        assert scene.is_dragging_wire_endpoint is True
        QTest.mouseMove(vp, _vpos(win, p_r3a[0] + 5, p_r3a[1] - 4))
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef(
            r3a.component.name, "a"
        )
        QTest.mouseRelease(
            vp,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _vpos(win, p_r3a[0] + 5, p_r3a[1] - 4),
        )
        assert scene.undo_stack.count() == undo_s3 + 2
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef(
            r3a.component.name, "a"
        )

        # Hand calculation: V_mid = 12 * 3000 / (1000 + 3000) = 9.0 V.
        conn_r3 = scene.extract_connectivity_result()
        sol_r3 = MNASolver(conn_r3.circuit).solve_dc()
        assert math.isclose(
            sol_r3.node_voltages[conn_r3.component_nodes["R1"]["b"]], 9.0, rel_tol=1e-6
        )

        # Undo/redo Acceptance S3 checks.
        scene.undo_stack.undo()
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == FreeEnd(120, -120)
        scene.undo_stack.undo()
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef("R2", "a")
        conn_u2 = scene.extract_connectivity_result()
        sol_u2 = MNASolver(conn_u2.circuit).solve_dc()
        assert math.isclose(
            sol_u2.node_voltages[conn_u2.component_nodes["R1"]["b"]], 6.0, rel_tol=1e-6
        )

        scene.undo_stack.redo()
        scene.undo_stack.redo()
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef(
            r3a.component.name, "a"
        )
        scene._wire_items_by_id[w_mid.wire_id].setSelected(True)

        # 3. Snap radius (<= 10 px snaps, 11 px misses) across all 8 orientations.
        rx = scene.place_new_component("resistor", QPointF(-200.0, 320.0))
        w_probe = scene.add_wire(FreeEnd(-360, 420), FreeEnd(-280, 420))
        assert w_probe is not None

        for rot in (0, 90, 180, 270):
            for flip in (False, True):
                rx.set_rotation_deg(rot)
                rx.set_flipped_h(flip)
                scene.refresh_connectivity_visuals()
                pa = rx.terminal_scene_positions_px()["a"]
                for dx, dy in ((0, 0), (1, 0), (3, 4), (8, 0), (9, 0)):
                    undo_n = scene.undo_stack.index()
                    QTest.mousePress(
                        vp,
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier,
                        _vpos(win, -280, 420),
                    )
                    QTest.mouseMove(vp, _vpos(win, pa[0] + dx, pa[1] + dy))
                    QTest.mouseRelease(
                        vp,
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier,
                        _vpos(win, pa[0] + dx, pa[1] + dy),
                    )
                    assert scene.wire_graph.wire(w_probe.wire_id).end_b == TerminalRef(
                        rx.component.name, "a"
                    )
                    assert scene.undo_stack.index() == undo_n + 1
                    scene.undo_stack.undo()
                    assert scene.undo_stack.index() == undo_n

                # 11 px away -> misses snap and lands on FreeEnd.
                undo_miss = scene.undo_stack.index()
                QTest.mousePress(
                    vp,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                    _vpos(win, -280, 420),
                )
                QTest.mouseMove(vp, _vpos(win, pa[0] + 11, pa[1]))
                QTest.mouseRelease(
                    vp,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                    _vpos(win, pa[0] + 11, pa[1]),
                )
                assert isinstance(
                    scene.wire_graph.wire(w_probe.wire_id).end_b, FreeEnd
                )
                assert scene.undo_stack.index() == undo_miss + 1
                scene.undo_stack.undo()
                assert scene.undo_stack.index() == undo_miss

        # 4. Rejections & cancel (0 undo steps, original wire preserved).
        scene._wire_items_by_id[w_mid.wire_id].setSelected(True)
        undo_rej = scene.undo_stack.count()
        # 4a. Same wire end (drag w_mid.end_b onto R1.b == w_mid.end_a)
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r3a)
        )
        QTest.mouseMove(vp, _vpos(win, *p_r1b))
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r1b)
        )
        assert scene.undo_stack.count() == undo_rej
        assert "terminal to itself" in scene.last_status_notice.lower()
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef(
            r3a.component.name, "a"
        )

        # 4b. Duplicate wire: add wire R1.b <-> R2.a, then drag w_mid end_b onto R2.a -> rejected.
        w_dup = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("R2", "a"))
        assert w_dup is not None
        scene.clearSelection()
        scene._wire_items_by_id[w_mid.wire_id].setSelected(True)
        undo_dup = scene.undo_stack.count()
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r3a)
        )
        QTest.mouseMove(vp, _vpos(win, *p_r2a))
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r2a)
        )
        assert scene.undo_stack.count() == undo_dup
        assert "already exists" in scene.last_status_notice.lower()
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef(
            r3a.component.name, "a"
        )

        # 4c. Esc cancel & Right-click cancel mid-drag
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r3a)
        )
        QTest.mouseMove(vp, _vpos(win, 240, -140))
        assert scene.is_dragging_wire_endpoint is True
        QTest.keyClick(view, Qt.Key.Key_Escape)
        assert scene.is_dragging_wire_endpoint is False
        assert scene.undo_stack.count() == undo_dup
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef(
            r3a.component.name, "a"
        )

        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *p_r3a)
        )
        QTest.mouseMove(vp, _vpos(win, 240, -140))
        QTest.mousePress(
            vp, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, _vpos(win, 240, -140)
        )
        assert scene.is_dragging_wire_endpoint is False
        assert scene.undo_stack.count() == undo_dup
        ctx = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            _vpos(win, 240, -140),
            vp.mapToGlobal(_vpos(win, 240, -140)),
        )
        view.contextMenuEvent(ctx)
        assert ctx.isAccepted() is True

        # 4d. Dropping FreeEnd on interior of another wire emits junctions-next-update notice.
        assert w_top is not None and len(w_top.points) >= 2
        top_mid = (
            (w_top.points[0][0] + w_top.points[1][0]) // 2,
            (w_top.points[0][1] + w_top.points[1][1]) // 2,
        )
        QTest.mousePress(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, -280, 420)
        )
        QTest.mouseMove(vp, _vpos(win, *top_mid))
        QTest.mouseRelease(
            vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vpos(win, *top_mid)
        )
        assert "junctions come in the next update" in scene.last_status_notice.lower()
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()


def test_phase5_5_router_and_50_component_drag_step_benchmark(qapp) -> None:
    """Phase 5.5: router micro-benchmark and 50-component drag-step latency benchmark."""
    from circuitsim.canvas.scene import CircuitScene
    from circuitsim.canvas.symbol_renderer import SymbolLibrary

    # 1. Qt-free router benchmark: analytic (0/1/2-bend) + obstacle-avoiding A*.
    obstacles = (
        (100, -40, 180, 40),
        (240, -60, 320, 20),
        (160, 80, 260, 140),
    )
    analytic_cases = (
        ((0, 0), (400, 0), (1, 0), (-1, 0), ()),  # 0-bend straight
        ((0, 0), (120, 100), (1, 0), (0, -1), ()),  # 1-bend L
        ((0, 0), (200, 120), (1, 0), (1, 0), ()),  # 2-bend Z
    )
    astar_cases = (
        ((0, 0), (200, 0), (-1, 0), (-1, 0), obstacles[:1]),  # obstacle U/detour
        ((40, 0), (380, 0), (1, 0), (-1, 0), obstacles),  # multi-obstacle A*
    )
    # Warm-up 1 pass
    for p0, p1, d0, d1, obs in analytic_cases + astar_cases:
        wire_route(p0, p1, start_dir=d0, end_dir=d1, obstacles=obs)

    t_r0 = time.perf_counter()
    for i in range(100):
        p0, p1, d0, d1, obs = analytic_cases[i % len(analytic_cases)]
        pts = wire_route(p0, p1, start_dir=d0, end_dir=d1, obstacles=obs)
        assert len(pts) >= 2
    route_analytic_ms = ((time.perf_counter() - t_r0) * 1000.0) / 100.0
    max_analytic_ms = float(os.environ.get("OHMBENCH_ROUTE_AVG_MAX_MS", "2.0"))
    assert route_analytic_ms <= max_analytic_ms, (
        f"analytic route() avg {route_analytic_ms:.3f} ms exceeded {max_analytic_ms} ms"
    )

    t_a0 = time.perf_counter()
    for i in range(20):
        p0, p1, d0, d1, obs = astar_cases[i % len(astar_cases)]
        pts = wire_route(p0, p1, start_dir=d0, end_dir=d1, obstacles=obs)
        assert len(pts) >= 2
    route_astar_ms = ((time.perf_counter() - t_a0) * 1000.0) / 20.0
    max_astar_ms = float(os.environ.get("OHMBENCH_ROUTE_ASTAR_MAX_MS", "15.0"))
    assert route_astar_ms <= max_astar_ms, (
        f"A* route() avg {route_astar_ms:.3f} ms exceeded {max_astar_ms} ms"
    )

    # 2. 50-component schematic drag-step benchmark.
    scene = CircuitScene(SymbolLibrary())
    items = []
    # 5 rows x 10 cols = 50 resistors spaced comfortably on a 20 px grid
    for row in range(5):
        for col in range(10):
            comp = scene.place_new_component(
                "resistor", QPointF(col * 160.0, row * 120.0)
            )
            items.append(comp)
            if col > 0:
                prev = items[-2]
                scene.add_wire(
                    TerminalRef(prev.component.name, "b"),
                    TerminalRef(comp.component.name, "a"),
                )
    assert len(scene.placed_items) == 50
    assert len(scene.placed_wires) == 45

    target = items[25]  # row 2, col 5 (middle wired component)
    orig_x, orig_y = target.pos().x(), target.pos().y()
    scene.clearSelection()
    target.setSelected(True)

    # Warm-up 1 step so lazy caches / Qt font metrics are already warm
    scene.begin_component_move_tracking()
    target.setPos(orig_x + 20.0, orig_y)
    scene.commit_component_move_tracking()

    step_times_ms: list[float] = []
    cur_x = orig_x + 20.0
    scene.begin_component_move_tracking()
    for k in range(1, 6):
        t0 = time.perf_counter()
        target.setPos(cur_x + 20.0 * k, orig_y)
        qapp.processEvents()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        step_times_ms.append(elapsed_ms)
    scene.commit_component_move_tracking()

    median_step_ms = statistics.median(step_times_ms)
    max_step_ms = float(os.environ.get("OHMBENCH_DRAG_STEP_MAX_MS", "30.0"))
    assert median_step_ms <= max_step_ms, (
        f"50-part drag step median {median_step_ms:.2f} ms (samples={step_times_ms}) "
        f"exceeded {max_step_ms} ms"
    )


def test_phase6_1_golden_roundtrips_and_s1_s5_matrix(tmp_path, qapp) -> None:
    """Phase 6.1: S1-S5 matrix + Schema v2 save/load & bundled golden circuit round-trips."""
    from pathlib import Path
    from circuitsim.wiring.serializer import (
        CircuitDocumentV2,
        load_document_v2 as load_document,
        save_document_v2 as save_document,
    )

    settings = AppSettings(data_dir=tmp_path / "settings_61")
    win = MainWindow(settings=settings)
    win.resize(1100, 800)
    win.show()
    qapp.processEvents()
    scene = win.circuit_scene

    try:
        # Build base 12 V / 1 kOhm / 2 kOhm divider -> V_mid = 12 * 2k / (1k + 2k) = 8.0 V.
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 20.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -60.0))
        r2 = scene.place_new_component("resistor", QPointF(160.0, 20.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-160.0, 140.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})

        w_top = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
        w_mid = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("R2", "a"))
        w_bot = scene.add_wire(TerminalRef("R2", "b"), TerminalRef("V1", "neg"))
        w_gnd = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
        assert all(w is not None for w in (w_top, w_mid, w_bot, w_gnd))

        # S1: moving, rotating, and flipping keeps TerminalRef connectivity & 8.0 V DC solve.
        scene.move_components_by([r1], 60.0, -40.0)
        scene.clearSelection()
        r1.setSelected(True)
        scene.rotate_selected()
        scene.flip_selected()
        conn_s1 = scene.extract_connectivity_result()
        sol_s1 = MNASolver(conn_s1.circuit).solve_dc()
        assert math.isclose(
            sol_s1.node_voltages[conn_s1.component_nodes["R1"]["b"]], 8.0, rel_tol=1e-6
        )

        # S2: crossing wires without an explicit Junction remain electrically isolated.
        # Wire 1: V_aux(24V) -> R_load -> GND crossing horizontally at y = 260
        # Wire 2: Free branch crossing vertically at x = 40 from y = 200 to y = 320 -> V2(5V)
        v2 = scene.place_new_component("dc_voltage", QPointF(260.0, 260.0), rotation_deg=90)
        r_aux = scene.place_new_component("resistor", QPointF(140.0, 200.0))
        scene.edit_component_parameters(v2, {"dc_voltage": 24.0})
        scene.edit_component_parameters(r_aux, {"resistance": 1000.0})
        w_cross_h = scene.add_wire(
            FreeEnd(-40, 260), FreeEnd(120, 260), hints=()
        )
        w_cross_v = scene.add_wire(
            FreeEnd(40, 200), FreeEnd(40, 320), hints=()
        )
        assert w_cross_h is not None and w_cross_v is not None
        assert len(scene.wire_graph.junctions) == 0

        # S3: detach w_mid to FreeEnd and undo back in 1 undo step.
        assert w_mid is not None
        assert scene.begin_wire_endpoint_drag(scene._wire_items_by_id[w_mid.wire_id], "end_b")
        assert scene.commit_wire_endpoint_drag((180, -100))
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == FreeEnd(180, -100)
        assert w_mid.wire_id in scene.extract_connectivity_result().open_wire_ids
        scene.undo_stack.undo()
        assert scene.wire_graph.wire(w_mid.wire_id).end_b == TerminalRef("R2", "a")

        # S4: add hints & Schema v2 golden save/load round-trip preserves WireGraph + DC operating point.
        w_hinted = scene.add_wire(
            FreeEnd(220, -100),
            FreeEnd(320, -40),
            hints=((320, -100),),
        )
        assert w_hinted is not None
        scene.add_wire(TerminalRef("V2", "pos"), TerminalRef(r_aux.component.name, "a"))
        scene.add_wire(TerminalRef(r_aux.component.name, "b"), TerminalRef("V1", "neg"))
        scene.add_wire(TerminalRef("V2", "neg"), TerminalRef("V1", "neg"))

        # S5: degree-3 junction dot + automatic merge to degree-2 hint when 3rd branch deleted.
        assert w_mid is not None
        split_res = scene.wire_graph.split_wire_at(w_mid.wire_id, (100, -100))
        g_j3 = split_res.graph.add_wire(
            JunctionRef(split_res.junction_id), FreeEnd(100, -160)
        )
        scene.set_wire_graph(g_j3)
        assert split_res.junction_id in scene._junction_items_by_id
        # Remove the 3rd branch -> junction disappears and remaining pair keeps (100, -100) hint.
        branch_id = g_j3.wires[-1].id
        scene.set_wire_graph(g_j3.remove_wire(branch_id))
        assert split_res.junction_id not in scene._junction_items_by_id
        assert (100, -100) in scene.wire_graph.wire(split_res.first_wire_id).hints

        golden_path = tmp_path / "golden_v2_matrix.circ"
        doc_v2 = scene.to_document_v2()
        assert isinstance(doc_v2, CircuitDocumentV2)
        save_document(golden_path, doc_v2)

        loaded_doc = load_document(golden_path)
        assert isinstance(loaded_doc, CircuitDocumentV2)
        win2 = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings_61_b"))
        try:
            win2.circuit_scene.load_document(loaded_doc)
            assert win2.circuit_scene.wire_graph == scene.wire_graph
            conn_rt = win2.circuit_scene.extract_connectivity_result()
            sol_rt = MNASolver(conn_rt.circuit).solve_dc()
            assert math.isclose(
                sol_rt.node_voltages[conn_rt.component_nodes["R1"]["b"]], 8.0, rel_tol=1e-6
            )
            assert math.isclose(
                sol_rt.node_voltages[conn_rt.component_nodes["V2"]["pos"]], 24.0, rel_tol=1e-6
            )
        finally:
            win2.circuit_scene.undo_stack.setClean()
            win2.close()

        # Round-trip bundled examples through v2 serialization and compare DC solve before/after.
        examples_dir = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "circuitsim"
            / "resources"
            / "examples"
        )
        for ex_name in ("voltage_divider.circ", "series_sources.circ", "led_resistor.circ"):
            ex_doc = load_document(examples_dir / ex_name)
            scene.load_document(ex_doc)
            sol_before = MNASolver(scene.extract_connectivity_result().circuit).solve_dc()
            rt_path = tmp_path / f"rt_{ex_name}"
            save_document(rt_path, scene.to_document_v2())
            scene.load_document(load_document(rt_path))
            sol_after = MNASolver(scene.extract_connectivity_result().circuit).solve_dc()
            assert sol_after.node_voltages.keys() == sol_before.node_voltages.keys()
            for node, volt in sol_before.node_voltages.items():
                assert math.isclose(sol_after.node_voltages[node], volt, rel_tol=1e-9, abs_tol=1e-9)
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()

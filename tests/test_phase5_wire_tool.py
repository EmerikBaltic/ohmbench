"""Tests for Phase 5 (`Step 4A-2`): interactive wire drawing, waypoints, segment drag, and end editing."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.engine import MNASolver, SimulationSession, SimulationState
from circuitsim.wiring.model import FreeEnd, Junction, JunctionRef, TerminalRef, WireGraph
from circuitsim.wiring.router import slide_segment_hints


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
        # Esc while preview was active cancels preview but keeps W mode on.
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

        # 8. Ending on the interior of another wire (far from terminals/junctions) creates FreeEnd + status note in 5.1.
        w_target = scene.add_wire(FreeEnd(0, 260), FreeEnd(160, 260))
        assert w_target is not None
        wires_before = len(scene.wire_graph.wires)
        junctions_before = len(scene.wire_graph.junctions)

        scene.start_wire_preview((80, 320))
        w_on_wire = scene.finish_wire_preview((80, 260))
        assert w_on_wire is not None
        assert w_on_wire.end_b == FreeEnd(80, 260)
        # Target wire was NOT split and NO junction was created in 5.1.
        assert len(scene.wire_graph.wires) == wires_before + 1
        assert len(scene.wire_graph.junctions) == junctions_before
        expected_note = "Wire ends on another wire: junctions come in the next update"
        assert scene.last_status_notice == expected_note
        assert win.statusBar().currentMessage() == expected_note

        # Undo reverts w_on_wire in 1 undo step.
        scene.undo_stack.undo()
        assert len(scene.wire_graph.wires) == wires_before
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

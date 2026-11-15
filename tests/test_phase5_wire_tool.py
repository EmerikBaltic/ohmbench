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

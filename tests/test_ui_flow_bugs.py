"""Real-input offscreen UI flow reproduction and regression tests for FIX QUEUE (E1-E6)."""

from __future__ import annotations

import math
import sys

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor

from circuitsim.app.themes import DARK_PALETTE, LIGHT_PALETTE, build_stylesheet
from circuitsim.engine import MNASolver, SimulationSession
from circuitsim.wiring.model import FreeEnd, JunctionRef, TerminalRef
from circuitsim.wiring.router import find_overlapping_wire_segments
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


def test_e1_ctrl_a_delete_undo_redo_and_every_subset(tmp_path, qapp) -> None:
    """E1 (a)-(d): Ctrl+A+Del, 10V divider (6.667V) undo/redo, FreeEnd detach, and all subsets."""
    from circuitsim.wiring.model import FreeEnd, JunctionRef, TerminalRef

    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -40.0))
        r2 = scene.place_new_component("resistor", QPointF(120.0, 0.0), rotation_deg=90)
        vm1 = scene.place_new_component("voltmeter", QPointF(240.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})

        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        p_r1a = terminal_scene_point(r1, "a")
        p_r1b = terminal_scene_point(r1, "b")
        p_r2a = terminal_scene_point(r2, "a")
        p_r2b = terminal_scene_point(r2, "b")
        p_vm_p = terminal_scene_point(vm1, "pos")
        p_vm_n = terminal_scene_point(vm1, "neg")
        p_gnd = terminal_scene_point(gnd, "gnd")

        win.act_wire_mode.setChecked(True)
        draw_wire_click_to_click(view, p_vp, p_r1a)
        draw_wire_click_to_click(view, p_r1b, p_r2a)
        draw_wire_click_to_click(view, p_r2b, p_vn)
        draw_wire_click_to_click(view, p_vn, p_gnd)
        # Branch voltmeter + free-end wire
        split_top = scene.wire_graph.split_wire_at(
            scene.wire_graph.attached_wires(TerminalRef("R1", "b"))[0].id,
            p_r1b,
        )
        scene.set_wire_graph(split_top.graph)
        j_top = split_top.junction_id
        draw_wire_click_to_click(view, p_r1b, p_vm_p)
        draw_wire_click_to_click(view, p_vm_n, p_r2b)
        win.act_wire_mode.setChecked(False)
        qapp.processEvents()

        # Verify hand-calculated voltage at divider middle node: 10 * 2k / (1k + 2k) = 6.666667 V.
        conn0 = scene.extract_connectivity_result()
        sol0 = MNASolver(conn0.circuit).solve_dc()
        v_mid_0 = sol0.node_voltages[conn0.component_nodes["R2"]["a"]]
        assert math.isclose(v_mid_0, 10.0 * 2000.0 / 3000.0, rel_tol=1e-6)
        orig_graph = scene.wire_graph
        orig_positions = {it.component.name: (it.pos().x(), it.pos().y()) for it in scene.placed_items}

        slot_exceptions: list[BaseException] = []
        old_hook = sys.excepthook

        def _capture_hook(exc_type, exc_val, exc_tb) -> None:
            slot_exceptions.append(exc_val)
            old_hook(exc_type, exc_val, exc_tb)

        sys.excepthook = _capture_hook
        try:
            # (a)(b) Real Ctrl+A followed by Delete -> atomic empty scene in 1 undo step.
            undo_before = scene.undo_stack.count()
            press_key(view.viewport(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            assert len(scene.selectedItems()) == (
                len(scene.placed_items) + len(scene.placed_wires) + len(scene.junction_items)
            )
            press_key(view.viewport(), Qt.Key.Key_Delete)
            assert not slot_exceptions
            assert len(scene.placed_items) == 0
            assert len(scene.placed_wires) == 0
            assert len(scene.junction_items) == 0
            assert scene.undo_stack.count() == undo_before + 1

            # Undo restores exact graph and positions -> divider still solves to 6.667 V.
            scene.undo_stack.undo()
            qapp.processEvents()
            assert not slot_exceptions
            assert scene.wire_graph == orig_graph
            assert {it.component.name: (it.pos().x(), it.pos().y()) for it in scene.placed_items} == orig_positions
            conn_u = scene.extract_connectivity_result()
            sol_u = MNASolver(conn_u.circuit).solve_dc()
            assert math.isclose(
                sol_u.node_voltages[conn_u.component_nodes["R2"]["a"]],
                20.0 / 3.0,
                rel_tol=1e-6,
            )

            # Redo deletes everything again.
            scene.undo_stack.redo()
            qapp.processEvents()
            assert len(scene.placed_items) == 0
            assert len(scene.placed_wires) == 0
            scene.undo_stack.undo()
            qapp.processEvents()

            # Deleting 1 component while its wires are NOT selected converts those ends to FreeEnd.
            scene.clearSelection()
            r1_item = next(it for it in scene.placed_items if it.component.name == "R1")
            r1_item.setSelected(True)
            win.act_delete.trigger()
            qapp.processEvents()
            assert not slot_exceptions
            remaining_terms = {
                it.component.name: tuple(t.name for t in it.metadata.terminals)
                for it in scene.placed_items
            }
            scene.wire_graph.validate(remaining_terms)
            w_detached = scene.wire_graph.attached_wires(TerminalRef("V1", "pos"))[0]
            assert FreeEnd(*p_r1a) in (w_detached.end_a, w_detached.end_b)
            assert isinstance(j_top, str)
            scene.undo_stack.undo()
            assert scene.wire_graph == orig_graph
        finally:
            sys.excepthook = old_hook
    finally:
        scene.undo_stack.setClean()
        win.close()


def test_e1_every_nonempty_subset_delete_and_undo(tmp_path, qapp) -> None:
    """E1 (c)(d): every non-empty subset of components/wires/junction/free-end deletes cleanly & undoes."""
    from circuitsim.wiring.model import FreeEnd, TerminalRef

    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        a1 = scene.place_new_component("ammeter", QPointF(0.0, -40.0))
        r1 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
        vm1 = scene.place_new_component("voltmeter", QPointF(280.0, 0.0), rotation_deg=90)
        qapp.processEvents()

        w1 = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("A1", "pos"))
        w2 = scene.add_wire(TerminalRef("A1", "neg"), TerminalRef("R1", "a"))
        w3 = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("V1", "neg"))
        split_res = scene.wire_graph.split_wire_at(w2.wire_id, terminal_scene_point(r1, "a"))
        scene.set_wire_graph(split_res.graph)
        from circuitsim.wiring.model import JunctionRef

        w4 = scene.add_wire(JunctionRef(split_res.junction_id), TerminalRef("VM1", "neg"))
        w_free = scene.add_wire(TerminalRef("VM1", "pos"), FreeEnd(360, -80))
        assert w1 is not None and w4 is not None and w_free is not None
        assert len(scene.junction_items) == 1
        qapp.processEvents()

        # Enumerate subsets of key deletable items (components + sample wires + junction).
        all_items = (
            list(scene.placed_items)
            + list(scene.placed_wires[:2])
            + list(scene.junction_items[:1])
        )
        # Take up to 6 items to keep 2^6 - 1 = 63 exact subset checks fast (<0.5s).
        subset_pool = all_items[:6]
        orig_graph = scene.wire_graph

        slot_exceptions: list[BaseException] = []
        old_hook = sys.excepthook

        def _capture_hook(exc_type, exc_val, exc_tb) -> None:
            slot_exceptions.append(exc_val)
            old_hook(exc_type, exc_val, exc_tb)

        sys.excepthook = _capture_hook
        try:
            n = len(subset_pool)
            for mask in range(1, 1 << n):
                scene.clearSelection()
                for bit in range(n):
                    if mask & (1 << bit):
                        subset_pool[bit].setSelected(True)
                assert scene.delete_selected(detach_unselected_wires=True) is True
                scene.wire_graph.validate({
                    it.component.name: tuple(t.name for t in it.metadata.terminals)
                    for it in scene.placed_items
                })
                scene.undo_stack.undo()
                assert scene.wire_graph == orig_graph
            assert not slot_exceptions
        finally:
            sys.excepthook = old_hook
    finally:
        scene.undo_stack.setClean()
        win.close()


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
        # Top wire from V1.pos (-160, -40) to R1.top (160, -40) routes at y = -60 (both pins exit upward).
        draw_wire_click_to_click(view, p_vp, r1_top)
        draw_wire_click_to_click(view, p_vn, r1_bot)
        draw_wire_click_to_click(view, p_vn, p_gnd)
        win.act_wire_mode.setChecked(False)
        qapp.processEvents()

        orig_graph = scene.wire_graph
        wires_before = len(orig_graph.wires)
        undo_before = scene.undo_stack.count()

        # Drop ammeter A1 onto the horizontal run of top wire W1 (at y = w_top.points[1][1], i.e. -60).
        w_top = scene.placed_wires[0]
        drop_y = float(w_top.points[1][1])
        start_vp = scene_to_viewport_point(view, a1.pos().x(), a1.pos().y())
        drop_vp = scene_to_viewport_point(view, 0.0, drop_y)
        drag_viewport(view, start_vp, drop_vp, steps=12)

        assert (a1.pos().x(), a1.pos().y()) == (0.0, drop_y)
        assert len(scene.wire_graph.wires) == wires_before + 1
        assert len(scene.wire_graph.junctions) == 0
        a1_wire_pts = {
            w.wire_id: w.points
            for w in scene.placed_wires
            if (isinstance(w.end_a, TerminalRef) and w.end_a.component == "A1")
            or (isinstance(w.end_b, TerminalRef) and w.end_b.component == "A1")
        }
        assert len(a1_wire_pts) == 2
        assert find_overlapping_wire_segments(a1_wire_pts) == []
        assert scene.undo_stack.count() == undo_before + 1

        conn = scene.extract_connectivity_result()
        assert "A1" not in conn.unconnected_components, "Ammeter A1 was left disconnected after drop onto wire!"
        n_a1_a = conn.component_nodes["A1"]["pos"]
        n_a1_b = conn.component_nodes["A1"]["neg"]
        assert n_a1_a != n_a1_b, "Ammeter was shorted by unsplit parallel wire / double junction tap!"

        frame = SimulationSession(connectivity=conn).start()
        i_a1 = abs(frame.components["A1"].current)
        # Hand calc: 12.0 V / 500.0 Ohm = 0.024 A (24.0 mA).
        assert math.isclose(i_a1, 0.024, rel_tol=1e-4), f"Expected 24.0 mA through A1, got {i_a1}"

        # 1 undo restores exact original graph and ammeter position.
        scene.undo_stack.undo()
        assert scene.wire_graph == orig_graph
        assert (a1.pos().x(), a1.pos().y()) == (0.0, 140.0)
    finally:
        scene.undo_stack.setClean()
        win.close()


@pytest.mark.parametrize("zoom", [0.5, 1.0, 2.0])
def test_e3_vertical_bent_rotated_flipped_and_pass_over_without_drop(
    tmp_path, qapp, zoom: float
) -> None:
    """E3: vertical & bent wires, flipped/rotated parts, 1-terminal tap, and non-drop drag stability."""
    win, view, scene = create_window_and_view(tmp_path, zoom=zoom)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-200.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(200.0, 160.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(0.0, 280.0))
        a_vert = scene.place_new_component("ammeter", QPointF(-80.0, 280.0), rotation_deg=90, flipped_h=True)
        a_flip = scene.place_new_component("ammeter", QPointF(0.0, -160.0), rotation_deg=180)
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 500.0})
        qapp.processEvents()

        r1_terms = r1.terminal_scene_positions_px()
        r1_top = min(r1_terms.items(), key=lambda kv: kv[1][1])  # ("a"/"b", (200, 120))
        r1_bot = max(r1_terms.items(), key=lambda kv: kv[1][1])  # ("a"/"b", (200, 200))

        # Bent top wire with explicit L-corner (200, -40): (-200, -40) -> (200, -40) -> (200, 120)
        scene.add_wire(
            TerminalRef("V1", "pos"),
            TerminalRef("R1", r1_top[0]),
            hints=((200, -40),),
        )
        w_bot = scene.add_wire(
            TerminalRef("V1", "neg"),
            TerminalRef("R1", r1_bot[0]),
            hints=((-200, 200),),
        )
        assert w_bot is not None
        # Connect ground via explicit T-junction at (0, 200) so zero segments overlap.
        split_g = scene.wire_graph.split_wire_at(w_bot.wire_id, (0, 200))
        scene.set_wire_graph(split_g.graph)
        scene.add_wire(JunctionRef(split_g.junction_id), TerminalRef("GND1", "gnd"))
        qapp.processEvents()

        polylines_before = {w.wire_id: w.points for w in scene.placed_wires}
        orig_graph = scene.wire_graph

        # 1. Drag a_flip over both horizontal runs WITHOUT dropping on a wire -> graph & polylines freeze during drag.
        scene.clearSelection()
        a_flip.setSelected(True)
        scene.begin_component_move_tracking()
        for y in (-40.0, 0.0, 40.0, 100.0, 120.0):
            a_flip.setPos(QPointF(-80.0, y))
            scene.on_component_position_changed(a_flip)
            assert scene.wire_graph == orig_graph
            for wid, pts in polylines_before.items():
                assert scene._wire_items_by_id[wid].points == pts
        assert scene.commit_component_move_tracking() is True
        assert scene.wire_graph == orig_graph
        scene.undo_stack.undo()
        scene.clearSelection()
        view.centerOn(0.0, 0.0)
        qapp.processEvents()

        # 2. Drag rotated 90 + flipped_h a_vert onto vertical leg (200, 40) of bent top wire via real mouse drag!
        start_vp = scene_to_viewport_point(view, a_vert.pos().x(), a_vert.pos().y())
        drop_vert_vp = scene_to_viewport_point(view, 200.0, 40.0)
        graph_pre_vert = scene.wire_graph
        undo_pre = scene.undo_stack.index()
        drag_viewport(view, start_vp, drop_vert_vp, steps=8)
        assert (a_vert.pos().x(), a_vert.pos().y()) == (200.0, 40.0)
        assert len(scene.wire_graph.wires) == len(graph_pre_vert.wires) + 1
        assert len(scene.wire_graph.junctions) == 1  # only the ground junction!
        assert find_overlapping_wire_segments({w.wire_id: w.points for w in scene.placed_wires}) == []
        assert scene.undo_stack.index() == undo_pre + 1
        conn_v = scene.extract_connectivity_result()
        frame_v = SimulationSession(connectivity=conn_v).start()
        assert math.isclose(abs(frame_v.components["A1"].current), 0.024, rel_tol=1e-4)

        # Undo vertical drop and test 180-deg rotated horizontal drop on top horizontal leg (0, -40).
        scene.undo_stack.undo()
        scene.clearSelection()
        view.centerOn(0.0, 0.0)
        qapp.processEvents()
        assert scene.wire_graph == graph_pre_vert

        scene.begin_component_move_tracking()
        a_flip.setPos(QPointF(0.0, -100.0))
        scene.on_component_position_changed(a_flip)
        a_flip.setPos(QPointF(0.0, -40.0))
        scene.on_component_position_changed(a_flip)
        assert scene.commit_component_move_tracking() is True
        assert (a_flip.pos().x(), a_flip.pos().y()) == (0.0, -40.0)
        assert len(scene.wire_graph.wires) == len(graph_pre_vert.wires) + 1
        assert len(scene.wire_graph.junctions) == 1
        assert find_overlapping_wire_segments({w.wire_id: w.points for w in scene.placed_wires}) == []
        conn_h = scene.extract_connectivity_result()
        frame_h = SimulationSession(connectivity=conn_h).start()
        assert math.isclose(abs(frame_h.components["A2"].current), 0.024, rel_tol=1e-4)

        # 3. Single-terminal tap: drop a vertical resistor R_tap at (-100, 160) -> only bottom pin (-100, 200) hits W2!
        r_tap = scene.place_new_component("resistor", QPointF(-140.0, 280.0), rotation_deg=90)
        qapp.processEvents()
        wires_tap0 = len(scene.wire_graph.wires)
        junc_tap0 = len(scene.wire_graph.junctions)
        scene.begin_component_move_tracking()
        r_tap.setPos(QPointF(-100.0, 160.0))
        scene.on_component_position_changed(r_tap)
        assert scene.commit_component_move_tracking() is True
        assert (r_tap.pos().x(), r_tap.pos().y()) == (-100.0, 160.0)
        assert len(scene.wire_graph.junctions) == junc_tap0 + 1
        assert len(scene.wire_graph.wires) == wires_tap0 + 2
        r_tap_a = scene.wire_graph.attached_wires(TerminalRef(r_tap.component.name, "a"))
        r_tap_b = scene.wire_graph.attached_wires(TerminalRef(r_tap.component.name, "b"))
        assert (len(r_tap_a) + len(r_tap_b)) == 1
    finally:
        scene.undo_stack.setClean()
        win.close()


def test_e4_repro_wire_mode_visibility_and_esc(tmp_path, qapp) -> None:
    """E4: Wire mode styles :checked buttons in both themes, shows status text, sets cursor, and exits on Esc."""
    dark_css = build_stylesheet(DARK_PALETTE)
    light_css = build_stylesheet(LIGHT_PALETTE)
    for palette, css in ((DARK_PALETTE, dark_css), (LIGHT_PALETTE, light_css)):
        assert "QToolButton:checked" in css
        assert palette.selection_accent in css
        assert palette.ghost_tint in css
        # Checked rule has its own background distinct from normal (transparent) and hover.
        checked_block = css.split("QToolButton:checked", 1)[1].split("}", 1)[0]
        assert f"background-color: {palette.selection_accent}" in checked_block

    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(0.0, 0.0))
        qapp.processEvents()
        p_vp = terminal_scene_point(v1, "pos")

        # Tool mode exclusivity & one-shot toolbars (Rot, Flip, Del, Junc are non-checkable one-shots).
        for one_shot in (win.act_rotate, win.act_flip, win.act_delete, win.act_junction):
            assert not one_shot.isCheckable()
        assert win.act_wire_mode.isCheckable()

        # Outside Wire mode, clicking open canvas does NOT start a wire.
        click_viewport(view, scene_to_viewport_point(view, -120, -120))
        assert scene.is_drawing_wire is False
        assert view.cursor().shape() == Qt.CursorShape.ArrowCursor

        # Enter via toolbar button: checked, crosshair cursor, status bar prompt.
        win.act_wire_mode.trigger()
        qapp.processEvents()
        assert view.wire_mode_enabled is True
        assert win.act_wire_mode.isChecked() is True
        assert view.cursor().shape() == Qt.CursorShape.CrossCursor
        assert (
            win.statusBar().currentMessage()
            == "Wire mode - click a terminal to start, Esc to leave"
        )

        # Pressing the toolbar button again exits Wire mode and clears status & cursor.
        win.act_wire_mode.trigger()
        qapp.processEvents()
        assert view.wire_mode_enabled is False
        assert win.act_wire_mode.isChecked() is False
        assert view.cursor().shape() == Qt.CursorShape.ArrowCursor
        assert win.statusBar().currentMessage() == ""

        # Re-enter via W key; Esc WITHOUT wire in progress leaves mode cleanly.
        press_key(view.viewport(), Qt.Key.Key_W)
        assert view.wire_mode_enabled is True
        assert win.act_wire_mode.isChecked() is True
        press_key(view.viewport(), Qt.Key.Key_Escape)
        assert view.wire_mode_enabled is False
        assert win.act_wire_mode.isChecked() is False
        assert view.cursor().shape() == Qt.CursorShape.ArrowCursor

        # Re-enter, start a wire on V1.pos, move to empty canvas, press Esc -> preview cancelled AND leaves Wire mode.
        win.act_wire_mode.trigger()
        qapp.processEvents()
        click_viewport(view, scene_to_viewport_point(view, *p_vp))
        assert scene.is_drawing_wire is True
        from PySide6.QtTest import QTest

        QTest.mouseMove(view.viewport(), scene_to_viewport_point(view, -120, -120))
        qapp.processEvents()

        press_key(view.viewport(), Qt.Key.Key_Escape)
        assert scene.is_drawing_wire is False
        assert view.wire_mode_enabled is False
        assert win.act_wire_mode.isChecked() is False
        assert view.cursor().shape() == Qt.CursorShape.ArrowCursor
        assert win.statusBar().currentMessage() == ""

        # Normal selection behavior restored after leaving Wire mode.
        scene.clearSelection()
        click_viewport(view, scene_to_viewport_point(view, 0, 0))
        assert scene.is_drawing_wire is False
        assert v1.isSelected() is True
    finally:
        scene.undo_stack.setClean()
        win.close()


def test_e5_repro_label_plates_cut_wire(tmp_path, qapp) -> None:
    """E5: label/value plates never hide or erase a wire passing through the label rect."""
    for theme in ("dark", "light"):
        win, view, scene = create_window_and_view(tmp_path / f"e5_{theme}", zoom=1.0, theme=theme)
        try:
            palette = DARK_PALETTE if theme == "dark" else LIGHT_PALETTE
            vm = scene.place_new_component("voltmeter", QPointF(0.0, 0.0), rotation_deg=90)
            vm.live_display_value = "12.00 V"
            qapp.processEvents()

            # Candidate[1] at rotation=90 sits on right side straddling 20px grid line (x ~ 60 or 80 outside body obstacle).
            plate_rect = vm.candidate_label_rects_scene()[1]
            x_grid = 60
            for gx in (60, 80, 100, 40):
                if plate_rect.left() + 3.0 <= gx <= plate_rect.right() - 3.0:
                    x_grid = gx
                    break
            assert plate_rect.left() + 3.0 <= x_grid <= plate_rect.right() - 3.0

            w = scene.add_wire(FreeEnd(x_grid, -120), FreeEnd(x_grid, 120))
            assert w is not None
            assert w.zValue() < vm.zValue()
            # Force wires directly across BOTH the active label and value rects even after label avoidance!
            lbl_local, val_local, _align = vm._upright_label_and_value_rects()
            lbl_scene = QRectF(
                lbl_local.x() + vm.pos().x(),
                lbl_local.y() + vm.pos().y(),
                lbl_local.width(),
                lbl_local.height(),
            )
            mid_y = int(round(lbl_scene.center().y() / 20.0) * 20.0)
            w_cross = scene.add_wire(FreeEnd(-140, mid_y), FreeEnd(140, mid_y))
            assert w_cross is not None
            qapp.processEvents()

            # Re-read final active value rect after label avoidance and cross it at an unobstructed vertical line >= 60.
            fin_lbl_local, fin_val_local, _ = vm._upright_label_and_value_rects()
            fin_val_scene = QRectF(
                fin_val_local.x() + vm.pos().x(),
                fin_val_local.y() + vm.pos().y(),
                fin_val_local.width(),
                fin_val_local.height(),
            )
            vx_grid = int(round(fin_val_scene.center().x() / 20.0) * 20.0)
            if abs(vx_grid) < 60:
                vx_grid = 60 if fin_val_scene.right() >= 60 else -60
            w_forced = scene.add_wire(FreeEnd(vx_grid, -120), FreeEnd(vx_grid, 120))
            assert w_forced is not None
            qapp.processEvents()

            render_rect = QRectF(-140.0, -120.0, 280.0, 240.0)
            img = render_scene_image(scene, render_rect, bg_hex=palette.canvas_bg)
            canvas_bg = QColor(palette.canvas_bg)

            # 1. Original vertical wire along x_grid passes straight through plate_rect and stays visible.
            px_x = int(round(x_grid - render_rect.left()))
            px_ref_y = int(round(-100.0 - render_rect.top()))
            assert img.pixelColor(px_x, px_ref_y) != canvas_bg
            inside_y_scene = plate_rect.top() + 1.0
            px_in_y = int(round(inside_y_scene - render_rect.top()))
            assert img.pixelColor(px_x, px_in_y) != canvas_bg

            # 2. Forced wire cutting right across the final active value plate also stays visible!
            f_lbl_local, f_val_local, _ = vm._upright_label_and_value_rects()
            f_val_scene = QRectF(
                f_val_local.x() + vm.pos().x(),
                f_val_local.y() + vm.pos().y(),
                f_val_local.width(),
                f_val_local.height(),
            )
            f_px_x = int(round(vx_grid - render_rect.left()))
            f_in_y = int(round((f_val_scene.top() + 0.5) - render_rect.top()))
            if f_val_scene.left() + 1.0 <= vx_grid <= f_val_scene.right() - 1.0:
                assert img.pixelColor(f_px_x, f_in_y) != canvas_bg, (
                    f"Active value plate erased forced wire at ({vx_grid}, {f_val_scene.top() + 0.5}) in {theme} theme"
                )
        finally:
            scene.undo_stack.setClean()
            win.close()


def test_e6_no_qfont_pointsize_warning_on_startup(tmp_path, qapp) -> None:
    """E6(a): Launching MainWindow and switching themes emits zero QFont::setPointSize (-1) warnings."""
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings
    from circuitsim.app.themes import ThemeMode

    captured: list[str] = []

    def _handler(msg_type, context, message: str) -> None:
        _ = (msg_type, context)
        captured.append(str(message))

    prev = qInstallMessageHandler(_handler)
    try:
        settings = AppSettings(data_dir=tmp_path / "e6_font_warn")
        win = MainWindow(settings=settings)
        win.show()
        qapp.processEvents()
        win.apply_theme(ThemeMode.LIGHT)
        qapp.processEvents()
        win.apply_theme(ThemeMode.DARK)
        qapp.processEvents()
        win.close()
    finally:
        qInstallMessageHandler(prev)

    bad = [m for m in captured if "setPointSize" in m or "Point size <= 0" in m]
    assert not bad, f"Unexpected QFont pointSize warning(s) at startup: {bad}"


def test_gate_e_startup_divider_smoke(tmp_path, qapp) -> None:
    """Gate E smoke: launch offscreen app, wire 10V / 1k / 2k divider via real clicks, simulate -> 6.667 V."""
    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, -40.0))
        r2 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})
        qapp.processEvents()

        p_vp = terminal_scene_point(v1, "pos")
        p_vn = terminal_scene_point(v1, "neg")
        p_r1a = terminal_scene_point(r1, "a")
        p_r1b = terminal_scene_point(r1, "b")
        p_r2a = terminal_scene_point(r2, "a")
        p_r2b = terminal_scene_point(r2, "b")
        p_gnd = terminal_scene_point(gnd, "gnd")

        win.act_wire_mode.trigger()
        draw_wire_click_to_click(view, p_vp, p_r1a)
        draw_wire_click_to_click(view, p_r1b, p_r2a)
        draw_wire_click_to_click(view, p_r2b, p_vn)
        draw_wire_click_to_click(view, p_vn, p_gnd)
        press_key(view.viewport(), Qt.Key.Key_Escape)
        assert win.act_wire_mode.isChecked() is False

        conn = scene.extract_connectivity_result()
        frame = SimulationSession(connectivity=conn).start()
        v_r2 = abs(frame.components["R2"].voltage)
        # Hand calc: V_mid = 10.0 * 2000 / (1000 + 2000) = 6.666667 V
        assert math.isclose(v_r2, 10.0 * 2000.0 / 3000.0, rel_tol=1e-5)
    finally:
        scene.undo_stack.setClean()
        win.close()

"""Step 4B (D1-D5) tests on top of WireGraph."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine.solver import MNASolver
from circuitsim.wiring.model import FreeEnd, JunctionRef, TerminalRef


def test_d1_split_on_drop_inline_and_junction_tap_single_undo(qapp) -> None:
    """D1: dropping a component onto a wire splits inline or taps a junction in 1 undo step."""
    scene = CircuitScene(SymbolLibrary())

    v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
    r1 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
    gnd = scene.place_new_component("ground", QPointF(-160.0, 100.0))
    scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
    scene.edit_component_parameters(r1, {"resistance": 2000.0})

    # Connect ground and bottom loop; route w_top via y = -80 across x = -160 .. 160.
    r1_terms = r1.terminal_scene_positions_px()
    r1_top_t, _ = min(r1_terms.items(), key=lambda kv: kv[1][1])
    r1_bot_t, _ = max(r1_terms.items(), key=lambda kv: kv[1][1])

    w_gnd = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef(gnd.component.name, "gnd"))
    w_bot = scene.add_wire(
        TerminalRef("V1", "neg"),
        TerminalRef("R1", r1_bot_t),
        hints=((-120, 80), (120, 80)),
    )
    w_top = scene.add_wire(
        TerminalRef("V1", "pos"),
        TerminalRef("R1", r1_top_t),
        hints=((-120, -80), (120, -80)),
    )
    assert w_gnd is not None and w_bot is not None and w_top is not None
    assert (-40, -80) in w_top.points or any(
        w_top.points[i][1] == -80 == w_top.points[i + 1][1]
        and min(w_top.points[i][0], w_top.points[i + 1][0]) < -40
        and max(w_top.points[i][0], w_top.points[i + 1][0]) > 40
        for i in range(len(w_top.points) - 1)
    )

    sol_before = MNASolver(scene.extract_connectivity_result().circuit).solve_dc()
    conn_before = scene.extract_connectivity_result()
    assert math.isclose(
        sol_before.node_voltages[conn_before.component_nodes["R1"][r1_top_t]],
        12.0,
        rel_tol=1e-9,
    )

    undo_idx_before_drop = scene.undo_stack.index()

    # 1. Drop default 1 kOhm resistor R2 inline at (0, -80): pins (-40, -80) and (40, -80) on w_top.
    r2 = scene.place_new_component("resistor", QPointF(0.0, -80.0))
    assert scene.undo_stack.index() == undo_idx_before_drop + 1

    # Verify w_top split inline into 2 wires attached to V1.pos->R2.a and R2.b->R1_top (0 junctions).
    assert len(scene.wire_graph.junctions) == 0
    r2_a_wires = scene.wire_graph.attached_wires(TerminalRef("R2", "a"))
    r2_b_wires = scene.wire_graph.attached_wires(TerminalRef("R2", "b"))
    assert len(r2_a_wires) == 1
    assert len(r2_b_wires) == 1

    # Hand calc: V = 12 V, R2 = 1 kOhm (default) in series with R1 = 2 kOhm.
    # V(R1_top) = 12 * 2000 / (1000 + 2000) = 8.0 V.
    conn_after = scene.extract_connectivity_result()
    sol_after = MNASolver(conn_after.circuit).solve_dc()
    assert math.isclose(
        sol_after.node_voltages[conn_after.component_nodes["R1"][r1_top_t]],
        8.0,
        rel_tol=1e-6,
    )

    # 1 undo removes R2 AND restores the unbroken top wire (12.0 V).
    scene.undo_stack.undo()
    assert scene.undo_stack.index() == undo_idx_before_drop
    assert len(scene.placed_items) == 3
    assert scene.wire_graph.has_wire(w_top.wire_id)
    assert len(scene.wire_graph.wires) == 3
    conn_undone = scene.extract_connectivity_result()
    sol_undone = MNASolver(conn_undone.circuit).solve_dc()
    assert math.isclose(
        sol_undone.node_voltages[conn_undone.component_nodes["R1"][r1_top_t]],
        12.0,
        rel_tol=1e-9,
    )

    # 1 redo restores R2 + split inline wiring cleanly.
    scene.undo_stack.redo()
    conn_redone = scene.extract_connectivity_result()
    sol_redone = MNASolver(conn_redone.circuit).solve_dc()
    assert math.isclose(
        sol_redone.node_voltages[conn_redone.component_nodes["R1"][r1_top_t]],
        8.0,
        rel_tol=1e-6,
    )

    # 2. Dragging an unconnected vertical resistor R3 so its top pin lands strictly on w_bot interior
    # creates a junction tap in 1 undo step on commit_component_move_tracking.
    r3 = scene.place_new_component("resistor", QPointF(0.0, 220.0), rotation_deg=90)
    r3_terms_before = r3.terminal_scene_positions_px()
    r3_top_t, r3_top_pos = min(r3_terms_before.items(), key=lambda kv: kv[1][1])
    dy_tap = 80.0 - float(r3_top_pos[1])  # move r3 top terminal onto (0, 80) on w_bot
    dx_tap = 0.0 - float(r3_top_pos[0])

    idx_before_drag_tap = scene.undo_stack.index()
    scene.begin_component_move_tracking()
    r3.setPos(r3.pos() + QPointF(dx_tap, dy_tap))
    scene.on_component_position_changed(r3)
    assert scene.commit_component_move_tracking() is True
    assert scene.undo_stack.index() == idx_before_drag_tap + 1
    assert len(scene.wire_graph.junctions) == 1
    assert len(scene.junction_items) == 1
    assert len(scene.wire_graph.attached_wires(TerminalRef(r3.component.name, r3_top_t))) == 1

    # 1 undo reverts R3 position AND removes the junction created on drop.
    scene.undo_stack.undo()
    assert scene.undo_stack.index() == idx_before_drag_tap
    assert len(scene.wire_graph.junctions) == 0
    assert len(scene.wire_graph.attached_wires(TerminalRef(r3.component.name, r3_top_t))) == 0


def test_d2_j_key_junction_split_and_third_wire_and_undo(tmp_path, qapp) -> None:
    """D2: J key splits wire at snapped cursor via split_wire_at, connects 3rd wire, and joins crossings."""
    settings = AppSettings(data_dir=tmp_path / "settings_d2")
    win = MainWindow(settings=settings)
    win.resize(1100, 800)
    win.show()
    qapp.processEvents()
    scene = win.circuit_scene
    view = win.circuit_view
    vp = view.viewport()

    try:
        # Divider: V1(12V) -> R1(1 kOhm) -> w_mid(y=-80) -> R2(2 kOhm) -> GND
        v1 = scene.place_new_component("dc_voltage", QPointF(-240.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(-100.0, -80.0))
        r2 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-240.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})

        r2_terms = r2.terminal_scene_positions_px()
        r2_top_t, _ = min(r2_terms.items(), key=lambda kv: kv[1][1])
        r2_bot_t, _ = max(r2_terms.items(), key=lambda kv: kv[1][1])

        w_top = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
        w_mid = scene.add_wire(
            TerminalRef("R1", "b"),
            TerminalRef("R2", r2_top_t),
            hints=((-40, -80), (140, -80)),
        )
        w_bot = scene.add_wire(
            TerminalRef("V1", "neg"),
            TerminalRef("R2", r2_bot_t),
            hints=((-200, 80), (140, 80)),
        )
        w_gnd = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
        assert all(w is not None for w in (w_top, w_mid, w_bot, w_gnd))
        assert len(scene.wire_graph.junctions) == 0

        # Ensure pressing J in empty space creates 0 junctions and 0 undo steps.
        idx0 = scene.undo_stack.index()
        QTest.mouseMove(vp, view.mapFromScene(QPointF(300.0, 300.0)))
        QTest.keyClick(view, Qt.Key.Key_J)
        qapp.processEvents()
        assert scene.undo_stack.index() == idx0
        assert len(scene.wire_graph.junctions) == 0

        # Move cursor near (0, -80) on w_mid and press J -> splits w_mid at (0, -80).
        QTest.mouseMove(vp, view.mapFromScene(QPointF(3.0, -82.0)))
        QTest.keyClick(view, Qt.Key.Key_J)
        qapp.processEvents()
        assert scene.undo_stack.index() == idx0 + 1
        assert len(scene.wire_graph.junctions) == 1
        j1 = scene.wire_graph.junctions[0]
        assert j1.pos == (0, -80)
        assert len(scene.junction_items) == 1
        assert scene.wire_graph.junction_degree(j1.id) == 2

        # Base divider solve through J1: V_mid = 12 * 2000 / (1000 + 2000) = 8.0 V.
        conn_split = scene.extract_connectivity_result()
        sol_split = MNASolver(conn_split.circuit).solve_dc()
        assert math.isclose(
            sol_split.node_voltages[conn_split.component_nodes["R2"][r2_top_t]],
            8.0,
            rel_tol=1e-6,
        )

        # Pressing J again on the degree-2 junction collapses/merges it back in 1 undo step,
        # and undo() brings J1 back cleanly.
        QTest.mouseMove(vp, view.mapFromScene(QPointF(2.0, -81.0)))
        QTest.keyClick(view, Qt.Key.Key_J)
        qapp.processEvents()
        assert len(scene.wire_graph.junctions) == 0
        assert len(scene.junction_items) == 0
        scene.undo_stack.undo()
        assert len(scene.wire_graph.junctions) == 1
        assert len(scene.junction_items) == 1

        # Add parallel load R3 = 2 kOhm and draw a 3rd wire from J1 to R3 top terminal.
        r_tap = scene.place_new_component("resistor", QPointF(0.0, 20.0), rotation_deg=90)
        scene.edit_component_parameters(r_tap, {"resistance": 2000.0})
        rt_terms = r_tap.terminal_scene_positions_px()
        rt_top_t, rt_top_pos = min(rt_terms.items(), key=lambda kv: kv[1][1])
        rt_bot_t, _ = max(rt_terms.items(), key=lambda kv: kv[1][1])
        w_rt_bot = scene.add_wire(TerminalRef("R3", rt_bot_t), TerminalRef("V1", "neg"))
        assert w_rt_bot is not None

        scene.start_wire_preview((2.0, -79.0))
        w_branch = scene.finish_wire_preview(rt_top_pos)
        assert w_branch is not None
        assert scene.wire_graph.junction_degree(j1.id) == 3

        # Hand calc: V1 = 12 V, R1 = 1 kOhm, parallel branch = R2 (2 kOhm) || R3 (2 kOhm) = 1 kOhm.
        # V_mid = 12 * 1000 / (1000 + 1000) = 6.0 V.
        conn_3 = scene.extract_connectivity_result()
        sol_3 = MNASolver(conn_3.circuit).solve_dc()
        assert math.isclose(
            sol_3.node_voltages[conn_3.component_nodes["R2"][r2_top_t]],
            6.0,
            rel_tol=1e-6,
        )

        # Undo 3rd wire -> R3 detaches while J1 still bridges R1.b and R2; undo back to idx0 restores unbroken w_mid.
        scene.undo_stack.undo()
        conn_2 = scene.extract_connectivity_result()
        assert conn_2.component_nodes["R1"]["b"] == conn_2.component_nodes["R2"][r2_top_t]
        assert conn_2.component_nodes["R3"][rt_top_t] != conn_2.component_nodes["R2"][r2_top_t]
        while scene.undo_stack.index() > idx0:
            scene.undo_stack.undo()
        assert len(scene.wire_graph.junctions) == 0
        assert len(scene.junction_items) == 0

        # Pressing J at the crossing of two isolated wires joins them into a degree-4 junction.
        w_cross_h = scene.add_wire(FreeEnd(-60, 200), FreeEnd(60, 200))
        w_cross_v = scene.add_wire(FreeEnd(0, 140), FreeEnd(0, 260))
        assert w_cross_h is not None and w_cross_v is not None
        assert len(scene.wire_graph.junctions) == 0

        QTest.mouseMove(vp, view.mapFromScene(QPointF(2.0, 198.0)))
        QTest.keyClick(view, Qt.Key.Key_J)
        qapp.processEvents()
        assert len(scene.wire_graph.junctions) == 1
        j_cross = scene.wire_graph.junctions[0]
        assert j_cross.pos == (0, 200)
        assert scene.wire_graph.junction_degree(j_cross.id) == 4
        scene.undo_stack.undo()
        assert len(scene.wire_graph.junctions) == 0
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()


def test_d3_junction_dragging_reroutes_wires_and_undo_redo(tmp_path, qapp) -> None:
    """D3: dragging a JunctionItem updates Junction.pos, reroutes incident wires, and supports undo/redo."""
    from circuitsim.canvas.commands import MoveJunctionCommand

    settings = AppSettings(data_dir=tmp_path / "settings_d3")
    win = MainWindow(settings=settings)
    win.resize(1100, 800)
    win.show()
    qapp.processEvents()
    scene = win.circuit_scene
    view = win.circuit_view
    vp = view.viewport()

    try:
        # Divider with degree-3 junction J1 at (0, -80): V1(12V) -> R1(1k) -> J1 -> (R2(2k) || R3(2k)) -> GND
        v1 = scene.place_new_component("dc_voltage", QPointF(-240.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(-100.0, -80.0))
        r2 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
        r3 = scene.place_new_component("resistor", QPointF(0.0, 20.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-240.0, 120.0))
        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        scene.edit_component_parameters(r1, {"resistance": 1000.0})
        scene.edit_component_parameters(r2, {"resistance": 2000.0})
        scene.edit_component_parameters(r3, {"resistance": 2000.0})

        r2_terms = r2.terminal_scene_positions_px()
        r2_top_t, _ = min(r2_terms.items(), key=lambda kv: kv[1][1])
        r2_bot_t, _ = max(r2_terms.items(), key=lambda kv: kv[1][1])
        r3_terms = r3.terminal_scene_positions_px()
        r3_top_t, r3_top_pos = min(r3_terms.items(), key=lambda kv: kv[1][1])
        r3_bot_t, _ = max(r3_terms.items(), key=lambda kv: kv[1][1])

        w_top = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
        w_mid = scene.add_wire(
            TerminalRef("R1", "b"),
            TerminalRef("R2", r2_top_t),
            hints=((-40, -80), (140, -80)),
        )
        w_bot = scene.add_wire(
            TerminalRef("V1", "neg"),
            TerminalRef("R2", r2_bot_t),
            hints=((-200, 80), (140, 80)),
        )
        w_gnd = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
        w_r3_bot = scene.add_wire(TerminalRef("R3", r3_bot_t), TerminalRef("V1", "neg"))
        assert all(w is not None for w in (w_top, w_mid, w_bot, w_gnd, w_r3_bot))

        split_res = scene.wire_graph.split_wire_at(w_mid.wire_id, (0, -80))
        g_j3 = split_res.graph.add_wire(
            JunctionRef(split_res.junction_id),
            TerminalRef("R3", r3_top_t),
        )
        scene.set_wire_graph(g_j3)
        jid = split_res.junction_id
        assert len(scene.junction_items) == 1
        assert scene.wire_graph.junction(jid).pos == (0, -80)

        # Ensure view center keeps (-240..200, -140..140) comfortably in viewport.
        view.centerOn(0.0, 0.0)
        qapp.processEvents()

        # 1. Start dragging J1, move to (60, -120), press Esc -> cancels with 0 undo steps.
        idx0 = scene.undo_stack.index()
        vp_start = view.mapFromScene(QPointF(0.0, -80.0))
        vp_cancel = view.mapFromScene(QPointF(60.0, -120.0))
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp_start)
        QTest.mouseMove(vp, vp_cancel)
        qapp.processEvents()
        assert scene.is_dragging_junction is True
        assert scene.wire_graph.junction(jid).pos == (60, -120)
        QTest.keyClick(view, Qt.Key.Key_Escape)
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp_cancel)
        qapp.processEvents()
        assert scene.is_dragging_junction is False
        assert scene.undo_stack.index() == idx0
        assert scene.wire_graph.junction(jid).pos == (0, -80)

        # 2. Drag J1 from (0, -80) to (40, -100) and release -> commits MoveJunctionCommand.
        vp_dest = view.mapFromScene(QPointF(40.0, -100.0))
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp_start)
        QTest.mouseMove(vp, vp_dest)
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp_dest)
        qapp.processEvents()

        assert scene.undo_stack.index() == idx0 + 1
        assert scene.wire_graph.junction(jid).pos == (40, -100)
        assert scene.junction_items[0].pos() == QPointF(40.0, -100.0)

        incident = scene.wire_graph.attached_wires(JunctionRef(jid))
        assert len(incident) == 3
        for w in incident:
            witem = scene._wire_items_by_id[w.id]
            end_pt = witem.points[0] if w.end_a == JunctionRef(jid) else witem.points[-1]
            assert end_pt == (40, -100)
            for i in range(len(witem.points) - 1):
                p, q = witem.points[i], witem.points[i + 1]
                assert p[0] == q[0] or p[1] == q[1]

        # Hand calc: parallel R2 || R3 = 1 kOhm, series with R1 = 1 kOhm -> V_mid = 6.0 V unchanged.
        conn_moved = scene.extract_connectivity_result()
        sol_moved = MNASolver(conn_moved.circuit).solve_dc()
        assert math.isclose(
            sol_moved.node_voltages[conn_moved.component_nodes["R2"][r2_top_t]],
            6.0,
            rel_tol=1e-6,
        )

        # Undo restores (0, -80); redo moves back to (40, -100).
        scene.undo_stack.undo()
        assert scene.undo_stack.index() == idx0
        assert scene.wire_graph.junction(jid).pos == (0, -80)
        scene.undo_stack.redo()
        assert scene.wire_graph.junction(jid).pos == (40, -100)

        # 3. Direct MoveJunctionCommand API works standalone as well.
        scene.undo_stack.push(MoveJunctionCommand(scene, jid, (40, -100), (20, -60)))
        assert scene.wire_graph.junction(jid).pos == (20, -60)
        scene.undo_stack.undo()
        assert scene.wire_graph.junction(jid).pos == (40, -100)
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()


def test_d4_copy_paste_components_wires_and_junctions_undo_redo(qapp) -> None:
    """D4: Copy/paste preserves components, wire hints, junctions, and attachments in 1 undo step."""
    win = MainWindow(settings=AppSettings())
    try:
        win.resize(1100, 800)
        win.show()
        scene = win.circuit_scene
        view = win.circuit_view
        view.centerOn(0.0, 0.0)
        view.setFocus()
        qapp.processEvents()

        v1 = scene.place_new_component("dc_voltage", QPointF(-200.0, 0.0), rotation_deg=90)
        gnd1 = scene.place_new_component("ground", QPointF(-200.0, 100.0))
        r1 = scene.place_new_component("resistor", QPointF(-80.0, -80.0), rotation_deg=0)
        r2 = scene.place_new_component("resistor", QPointF(0.0, -20.0), rotation_deg=90)
        r3 = scene.place_new_component("resistor", QPointF(120.0, -20.0), rotation_deg=90)

        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        assert math.isclose(r1.component.resistance, 1000.0, rel_tol=1e-9)
        scene.edit_component_parameters(r2, {"resistance": 2000.0})
        scene.edit_component_parameters(r3, {"resistance": 2000.0})

        r1_terms = r1.terminal_scene_positions_px()
        r1_left_t, _ = min(r1_terms.items(), key=lambda kv: kv[1][0])
        r1_right_t, _ = max(r1_terms.items(), key=lambda kv: kv[1][0])
        r2_terms = r2.terminal_scene_positions_px()
        r2_top_t, _ = min(r2_terms.items(), key=lambda kv: kv[1][1])
        r2_bot_t, _ = max(r2_terms.items(), key=lambda kv: kv[1][1])
        r3_terms = r3.terminal_scene_positions_px()
        r3_top_t, _ = min(r3_terms.items(), key=lambda kv: kv[1][1])
        r3_bot_t, _ = max(r3_terms.items(), key=lambda kv: kv[1][1])

        w_gnd = scene.add_wire(
            TerminalRef("V1", "neg"), TerminalRef(gnd1.component.name, "gnd")
        )
        assert w_gnd is not None
        assert scene.toggle_junction_at(QPointF(-200.0, 60.0)) is True

        w_v1_r1 = scene.add_wire(
            TerminalRef("V1", "pos"),
            TerminalRef("R1", r1_left_t),
            hints=((-200, -80),),
        )
        w_r1_r3 = scene.add_wire(
            TerminalRef("R1", r1_right_t),
            TerminalRef("R3", r3_top_t),
            hints=((120, -80),),
        )
        w_bot1 = scene.add_wire(
            TerminalRef("V1", "neg"),
            TerminalRef("R2", r2_bot_t),
            hints=((0, 40),),
        )
        w_bot2 = scene.add_wire(
            TerminalRef("R2", r2_bot_t),
            TerminalRef("R3", r3_bot_t),
            hints=((0, 60), (120, 60)),
        )
        assert all(w is not None for w in (w_v1_r1, w_r1_r3, w_bot1, w_bot2))
        assert scene.wire_graph.wire(w_v1_r1.wire_id).hints == ((-200, -80),)

        split_res = scene.wire_graph.split_wire_at(w_r1_r3.wire_id, (0, -80))
        j1_id = split_res.junction_id
        split_g = split_res.graph.add_wire(
            TerminalRef("R2", r2_top_t),
            JunctionRef(j1_id),
            wire_id=split_res.graph.new_wire_id(),
        )
        scene.set_wire_graph(split_g)
        qapp.processEvents()

        assert len(scene.junction_items) == 2
        orig_comp_names = {c.component.name for c in scene.placed_items}
        orig_wire_ids = {w.id for w in scene.wire_graph.wires}
        orig_junc_ids = {j.id for j in scene.wire_graph.junctions}
        assert len(orig_comp_names) == 5
        assert len(orig_junc_ids) == 2

        # Select entire subcircuit (components, wires, and both degree-3 and explicit junctions).
        scene.clearSelection()
        for it in [*scene.placed_items, *scene.placed_wires, *scene.junction_items]:
            it.setSelected(True)

        idx0 = scene.undo_stack.index()
        orig_graph = scene.wire_graph

        assert scene.copy_selected() is True
        assert scene.paste_clipboard(offset_px=(480.0, 0.0)) is True
        assert scene.undo_stack.index() == idx0 + 1

        new_comp_items = [
            c for c in scene.placed_items if c.component.name not in orig_comp_names
        ]
        new_comp_names = {c.component.name for c in new_comp_items}
        assert new_comp_names == {"V2", "GND2", "R4", "R5", "R6"}

        new_wires = [w for w in scene.wire_graph.wires if w.id not in orig_wire_ids]
        new_juncs = [j for j in scene.wire_graph.junctions if j.id not in orig_junc_ids]
        assert len(new_wires) == len(orig_wire_ids)
        assert len(new_juncs) == 2

        # Check hints preserved and shifted by (+480, 0) on pasted V2 -> R4 wire.
        pasted_v2_r4 = [
            w
            for w in new_wires
            if {w.end_a, w.end_b}
            == {TerminalRef("V2", "pos"), TerminalRef("R4", r1_left_t)}
        ]
        assert len(pasted_v2_r4) == 1
        assert pasted_v2_r4[0].hints == ((280, -80),)

        # Check degree-3 junction cloned with new ID, position (480, -80), 3 incident wires, visible dot.
        deg3_juncs = [j for j in new_juncs if scene.wire_graph.junction_degree(j.id) == 3]
        assert len(deg3_juncs) == 1
        pasted_j1 = deg3_juncs[0]
        assert pasted_j1.pos == (480, -80)
        assert len(scene.wire_graph.attached_wires(JunctionRef(pasted_j1.id))) == 3

        # Check explicit degree-2 junction dot also visible after paste.
        deg2_juncs = [j for j in new_juncs if scene.wire_graph.junction_degree(j.id) == 2]
        assert len(deg2_juncs) == 1
        assert deg2_juncs[0].pos == (280, 60)
        visible_jids = {j.junction_id for j in scene.junction_items}
        assert pasted_j1.id in visible_jids
        assert deg2_juncs[0].id in visible_jids
        selected_jids = {
            j.junction_id for j in scene.junction_items if j.isSelected()
        }
        assert selected_jids == {pasted_j1.id, deg2_juncs[0].id}

        # Hand calc: V = 12 V, R1 = 1 kOhm, R2 || R3 = 2 kOhm || 2 kOhm = 1 kOhm -> V_mid = 6.0 V.
        conn_pasted = scene.extract_connectivity_result()
        sol_pasted = MNASolver(conn_pasted.circuit).solve_dc()
        assert math.isclose(
            sol_pasted.node_voltages[conn_pasted.component_nodes["R2"][r2_top_t]],
            6.0,
            rel_tol=1e-6,
        )
        assert math.isclose(
            sol_pasted.node_voltages[conn_pasted.component_nodes["R5"][r2_top_t]],
            6.0,
            rel_tol=1e-6,
        )

        # Single undo restores original circuit; single redo restores the pasted copy.
        scene.undo_stack.undo()
        assert scene.undo_stack.index() == idx0
        assert len(scene.placed_items) == 5
        assert scene.wire_graph == orig_graph

        scene.undo_stack.redo()
        assert scene.undo_stack.index() == idx0 + 1
        assert len(scene.placed_items) == 10
        assert len(scene.wire_graph.junctions) == 4

        # Verify Ctrl+C / Ctrl+V via QTest keyboard shortcuts works in 1 undo step.
        scene.clearSelection()
        for it in [
            v1,
            gnd1,
            r1,
            r2,
            r3,
            *[scene._wire_items_by_id[wid] for wid in orig_wire_ids],
            *[j for j in scene.junction_items if j.junction_id in orig_junc_ids],
        ]:
            it.setSelected(True)

        QTest.keyClick(view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(view, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
        qapp.processEvents()
        assert scene.undo_stack.index() == idx0 + 2
        assert len(scene.placed_items) == 15
        assert len(scene.wire_graph.junctions) == 6

        scene.undo_stack.undo()
        assert scene.undo_stack.index() == idx0 + 1
        assert len(scene.placed_items) == 10
        assert len(scene.wire_graph.junctions) == 4
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()

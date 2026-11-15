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

"""Tests for Phase 4 (`Step 4A-1`) integration of `circuitsim.wiring` into `CircuitScene`."""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter

from circuitsim.app.themes import ThemeMode, get_theme_palette
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine import MNASolver, SimulationSession, SimulationState
from circuitsim.wiring.model import FreeEnd, JunctionRef, TerminalRef
from circuitsim.wiring.router import route as wire_route


def test_phase4_1_scene_wire_graph_add_wire_attachments_and_undo_redo(qapp) -> None:
    """CircuitScene is backed by WireGraph and supports attachment + point add_wire with undo/redo."""
    scene = CircuitScene(SymbolLibrary())
    v1 = scene.place_new_component("dc_voltage", QPointF(0.0, 0.0))
    r1 = scene.place_new_component("resistor", QPointF(0.0, -80.0))
    gnd = scene.place_new_component("ground", QPointF(40.0, 40.0))
    scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
    scene.edit_component_parameters(r1, {"resistance": 1000.0})

    w1 = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    w2 = scene.add_wire([(40, 0), (40, -80)])
    w3 = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef(gnd.component.name, "gnd"))
    w4 = scene.add_wire(TerminalRef("V1", "neg"), FreeEnd(120, 20))

    assert w1 is not None and w2 is not None and w3 is not None and w4 is not None
    assert len(scene.wire_graph.wires) == 4
    assert w1.end_a == TerminalRef("V1", "pos") and w1.end_b == TerminalRef("R1", "a")
    assert w2.end_a == TerminalRef("V1", "neg") and w2.end_b == TerminalRef("R1", "b")
    assert w4.end_b == FreeEnd(120, 20)

    # Duplicate attachment wire is ignored and returns the existing WireItem.
    dup = scene.add_wire(TerminalRef("R1", "a"), TerminalRef("V1", "pos"))
    assert dup is w1
    assert len(scene.wire_graph.wires) == 4

    # Undo/redo updates scene.wire_graph and scene.placed_wires consistently.
    scene.undo_stack.undo()
    assert len(scene.wire_graph.wires) == 3
    assert len(scene.placed_wires) == 3
    scene.undo_stack.redo()
    assert len(scene.wire_graph.wires) == 4
    assert len(scene.placed_wires) == 4

    # Hand-calculated DC loop current: I = 10 V / 1000 ohm = 0.01 A
    circuit = scene.build_circuit()
    MNASolver(circuit).solve_dc()
    assert math.isclose(abs(circuit.get_component("R1").current), 0.01, rel_tol=1e-9)


def test_phase4_1_move_merge_and_wire_identity_and_split_junction(qapp) -> None:
    """Consecutive moves of the same components merge; delete preserves unrelated WireItem identity; point split creates Junction."""
    scene = CircuitScene(SymbolLibrary())
    v1 = scene.place_new_component("dc_voltage", QPointF(0.0, 0.0))
    r1 = scene.place_new_component("resistor", QPointF(0.0, -80.0))
    scene.place_new_component("ground", QPointF(40.0, 40.0))
    scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
    scene.edit_component_parameters(r1, {"resistance": 2000.0})

    w1 = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    w_free = scene.add_wire([(200, 0), (200, 120)])
    assert w1 is not None and w_free is not None

    # Tapping strictly inside w_free splits it in WireGraph at a JunctionRef.
    w_tap = scene.add_wire([(200, 60), (260, 60)])
    assert w_tap is not None
    assert len(scene.wire_graph.junctions) == 1
    assert isinstance(w_tap.end_a, JunctionRef)

    undo_before_moves = scene.undo_stack.index()
    assert scene.move_components_by([r1], 40.0, 0.0) is True
    assert scene.move_components_by([r1], 0.0, -40.0) is True
    assert r1.pos() == QPointF(40.0, -120.0)
    assert scene.undo_stack.index() == undo_before_moves + 1, "Consecutive moves of same selection should merge"

    # Single undo returns R1 to initial position and keeps WireItem object identity.
    scene.undo_stack.undo()
    assert r1.pos() == QPointF(0.0, -80.0)
    assert w1.points[-1] == (-40, -80)
    assert any(w is w_tap for w in scene.placed_wires)

    # Deleting R1 removes w1 via WireGraph.remove_component while keeping w_tap identical.
    scene.clearSelection()
    r1.setSelected(True)
    assert scene.delete_selected() is True
    assert not scene.wire_graph.has_wire(w1.wire_id)
    assert any(w is w_tap for w in scene.placed_wires)
    scene.undo_stack.undo()
    assert scene.wire_graph.has_wire(w1.wire_id)


def test_phase4_2_wire_item_routed_from_router_junction_dot_and_free_end(qapp) -> None:
    """WireItems use wiring.router.route(), JunctionItems track degree>=3, FreeEnd draws hollow markers, and hitbox is 10 px."""
    scene = CircuitScene(SymbolLibrary())
    r1 = scene.place_new_component("resistor", QPointF(0.0, 0.0))
    r2 = scene.place_new_component("resistor", QPointF(160.0, 80.0))
    r3 = scene.place_new_component("resistor", QPointF(160.0, -80.0))

    # Connect R1.b (40, 0) [outward (+1, 0)] to R2.a (120, 80) [outward (-1, 0)].
    w12 = scene.add_wire(TerminalRef(r1.component.name, "b"), TerminalRef(r2.component.name, "a"))
    assert w12 is not None
    expected_pts = wire_route(
        (40, 0),
        (120, 80),
        start_dir=(1, 0),
        end_dir=(-1, 0),
        obstacles=scene._component_obstacles_px(),
    )
    assert w12.points == expected_pts
    assert len(scene.junction_items) == 0

    # Verify 10 px hit stroke (half-width 5 px): 4 px away is inside shape, 6 px away is outside.
    first_x, first_y = w12.points[0]
    second_x, second_y = w12.points[1]
    assert first_y == second_y  # leaves R1.b horizontally
    mid_x = (first_x + second_x) / 2.0
    assert w12.shape().contains(QPointF(mid_x, float(first_y) + 4.0))
    assert not w12.shape().contains(QPointF(mid_x, float(first_y) + 6.0))

    # Tap interior of w12 at a grid bend/segment to create a 3-way junction -> JunctionItem appears.
    tap_x = w12.points[1][0]
    tap_y = (w12.points[1][1] + w12.points[2][1]) // 2
    tap_y = (tap_y // 20) * 20
    w_tap = scene.add_wire([(tap_x, tap_y), r3.terminal_scene_positions_px()["a"]])
    assert w_tap is not None
    assert isinstance(w_tap.end_a, JunctionRef)
    assert len(scene.junction_items) == 1
    jitem = scene.junction_items[0]
    assert jitem.junction_id == w_tap.end_a.id
    assert jitem.pos() == QPointF(float(tap_x), float(tap_y))

    # Removing the 3rd branch reduces/collapses the junction and removes JunctionItem.
    scene.clearSelection()
    w_tap.setSelected(True)
    assert scene.delete_selected() is True
    assert len(scene.junction_items) == 0

    # FreeEnd wire renders hollow circle marker on canvas in both themes without errors.
    w_open = scene.add_wire(TerminalRef(r1.component.name, "a"), FreeEnd(-120, 0))
    assert w_open is not None
    for theme in (ThemeMode.DARK, ThemeMode.LIGHT):
        scene.set_theme(theme)
        img = QImage(240, 120, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(get_theme_palette(theme).canvas_bg)
        painter = QPainter(img)
        scene.render(painter, QRectF(0, 0, 240, 120), QRectF(-160, -60, 240, 120))
        painter.end()
        # Marker ring at FreeEnd (-120, 0) corresponds to x=40, y=60; 3.5 px ring differs from canvas_bg.
        bg_rgb = img.pixelColor(2, 2).rgb()
        ring_pixels = [
            img.pixelColor(px, py).rgb()
            for px in range(36, 45)
            for py in range(56, 65)
        ]
        assert any(c != bg_rgb for c in ring_pixels)
        # Center of hollow ring matches canvas background (wire stops at (-120, 0), hollow inside left edge at 37..39).
        assert img.pixelColor(37, 60).rgb() != bg_rgb or img.pixelColor(43, 60).rgb() != bg_rgb


def test_phase4_3_netlist_from_wire_graph_and_open_wire_diagnostic(qapp) -> None:
    """Scene netlist uses wiring.netlist.build_circuit and reports amber open-wire warning for FreeEnd."""
    scene = CircuitScene(SymbolLibrary())
    v1 = scene.place_new_component("dc_voltage", QPointF(-100.0, 0.0), rotation_deg=90)
    r1 = scene.place_new_component("resistor", QPointF(0.0, -40.0))
    r2 = scene.place_new_component("resistor", QPointF(100.0, 0.0), rotation_deg=90)
    gnd = scene.place_new_component("ground", QPointF(-100.0, 60.0))

    scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
    scene.edit_component_parameters(r1, {"resistance": 1000.0})
    scene.edit_component_parameters(r2, {"resistance": 2000.0})

    # Wire up 10 V -> 1 kOhm -> 2 kOhm -> Ground
    # V1 rot=90: pos=(-100, -40), neg=(-100, 40); GND1 gnd=(-100, 40) (pin-to-pin contact!)
    w1 = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    w2 = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("R2", "a"))
    w3 = scene.add_wire(TerminalRef("R2", "b"), TerminalRef(gnd.component.name, "gnd"))
    assert w1 is not None and w2 is not None and w3 is not None

    # Add a wire with a FreeEnd crossing across the midpoint wire without a junction.
    # Also attach a separate resistor R3 on an independent FreeEnd branch to verify crossing does not short.
    r3 = scene.place_new_component("resistor", QPointF(220.0, -40.0))
    scene.edit_component_parameters(r3, {"resistance": 500.0})
    w_open = scene.add_wire(TerminalRef("R1", "b"), FreeEnd(160, -100))
    assert w_open is not None

    conn = scene.extract_connectivity_result()
    assert conn.open_wire_ids == [w_open.wire_id]

    session = SimulationSession(connectivity=conn)
    frame = session.start()
    assert frame.state == SimulationState.RUNNING

    # Hand calculation: V_mid = V1 * R2 / (R1 + R2) = 10.0 * 2000.0 / (1000.0 + 2000.0) = 20 / 3 V
    mid_node = conn.component_nodes["R2"]["a"]
    expected_v_mid = 10.0 * 2000.0 / (1000.0 + 2000.0)
    assert math.isclose(frame.node_voltages[mid_node], expected_v_mid, rel_tol=1e-6)

    # FreeEnd open wire produces a non-blocking amber warning diagnostic highlighted on w_open.
    open_diags = [d for d in frame.diagnostics if d.code == "open_wire"]
    assert len(open_diags) == 1
    assert open_diags[0].severity == "warning"
    assert w_open.wire_id in open_diags[0].wire_ids

    scene.apply_simulation_frame(frame)
    assert w_open.diagnostic_severity == "warning"
    assert w_open.effective_highlight_color() == "#FFA726"


def _net_partition(conn) -> frozenset[frozenset[tuple[str, str]]]:
    by_net: dict[str, set[tuple[str, str]]] = {}
    for cname, tmap in conn.component_nodes.items():
        for tname, net in tmap.items():
            by_net.setdefault(net, set()).add((cname, tname))
    return frozenset(frozenset(members) for members in by_net.values())


def _interior_grid_point(pts: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    best = None
    best_len = -1
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        length = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
        if length >= 40 and length > best_len:
            best_len = length
            best = (p1, p2)
    assert best is not None, f"No segment >= 40 px in {pts}"
    p1, p2 = best
    if p1[0] == p2[0]:
        mid_y = (((p1[1] + p2[1]) // 2) // 20) * 20
        return (p1[0], mid_y)
    mid_x = (((p1[0] + p2[0]) // 2) // 20) * 20
    return (mid_x, p1[1])


def test_phase4_4_acceptance_s1_s2_and_delete_junction_merge(qapp) -> None:
    """Acceptance S1 & S2: moving/rotating/flipping VM1 preserves attachment/reading,
    dropping VM1 onto R1 creates no new connection, and delete/merge/undo on JunctionItems works.
    """
    scene = CircuitScene(SymbolLibrary())
    v1 = scene.place_new_component("dc_voltage", QPointF(0.0, 0.0), rotation_deg=90)
    r1 = scene.place_new_component("resistor", QPointF(100.0, -40.0))
    r2 = scene.place_new_component("resistor", QPointF(180.0, 0.0), rotation_deg=90)
    gnd = scene.place_new_component("ground", QPointF(0.0, 120.0))
    vm1 = scene.place_new_component("voltmeter", QPointF(280.0, 0.0), rotation_deg=90)

    scene.edit_component_parameters(v1, {"dc_voltage": 10.0})
    scene.edit_component_parameters(r1, {"resistance": 1000.0})
    scene.edit_component_parameters(r2, {"resistance": 2000.0})
    scene.edit_component_parameters(vm1, {"is_ideal": True})

    w_top = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    w_mid = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("R2", "a"))
    w_bot = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("R2", "b"))
    assert w_bot is not None
    # Split w_bot at an interior grid point to connect ground via a degree-3 junction.
    w_gnd = scene.add_wire(
        [_interior_grid_point(w_bot.points), gnd.terminal_scene_positions_px()["gnd"]]
    )
    w_vm_pos = scene.add_wire(TerminalRef("R2", "a"), TerminalRef("VM1", "pos"))
    w_vm_neg = scene.add_wire(TerminalRef("R2", "b"), TerminalRef("VM1", "neg"))
    assert all(w is not None for w in (w_top, w_mid, w_bot, w_gnd, w_vm_pos, w_vm_neg))
    assert len(scene.junction_items) == 1

    # Hand calculation: V(R2) = 10.0 * 2000.0 / (1000.0 + 2000.0) = 20 / 3 V; V(R1) = 10 / 3 V
    expected_v_r2 = 10.0 * 2000.0 / (1000.0 + 2000.0)
    expected_v_r1 = 10.0 - expected_v_r2

    base_conn = scene.extract_connectivity_result()
    base_partition = _net_partition(base_conn)
    MNASolver(base_conn.circuit).solve_dc()
    assert math.isclose(base_conn.circuit.get_component("VM1").reading, expected_v_r2, rel_tol=1e-6)

    def _assert_divider_unchanged() -> None:
        tpos = vm1.terminal_scene_positions_px()
        assert w_vm_pos.points[0] == tpos["pos"] or w_vm_pos.points[-1] == tpos["pos"]
        assert w_vm_neg.points[0] == tpos["neg"] or w_vm_neg.points[-1] == tpos["neg"]
        conn = scene.extract_connectivity_result()
        assert _net_partition(conn) == base_partition
        sol = MNASolver(conn.circuit).solve_dc()
        vm = conn.circuit.get_component("VM1")
        r1_c = conn.circuit.get_component("R1")
        assert math.isclose(vm.reading, expected_v_r2, rel_tol=1e-6)
        assert math.isclose(abs(r1_c.voltage), expected_v_r1, rel_tol=1e-6)
        assert math.isclose(
            sol.node_voltages[conn.component_nodes["R2"]["a"]],
            expected_v_r2,
            rel_tol=1e-6,
        )

    # S1: move 40, 100, 300 px in all 4 directions, rotate 4x, and flip 2x
    for dist in (40.0, 100.0, 300.0):
        for dx, dy in ((dist, 0.0), (-dist, 0.0), (0.0, dist), (0.0, -dist)):
            assert scene.move_components_by([vm1], dx, dy, merge=False) is True
            _assert_divider_unchanged()

    # Also verify interactive drag-move tracking path keeps endpoints & netlist intact.
    scene.begin_component_move_tracking()
    vm1.setPos(vm1.pos() + QPointF(40.0, -40.0))
    scene.on_component_position_changed(vm1)
    assert scene.commit_component_move_tracking() is True
    _assert_divider_unchanged()

    scene.clearSelection()
    vm1.setSelected(True)
    for _ in range(4):
        assert scene.rotate_selected() is True
        _assert_divider_unchanged()
    for _ in range(2):
        assert scene.flip_selected() is True
        _assert_divider_unchanged()

    # S2: drag VM1 onto R1's terminals (and 1 grid step near them) -> NO new connection created
    while vm1.rotation_deg != 0:
        scene.rotate_selected()
    r1_a = r1.terminal_scene_positions_px()["a"]
    vm_pos = vm1.terminal_scene_positions_px()["pos"]
    assert scene.move_components_by(
        [vm1], float(r1_a[0] - vm_pos[0]), float(r1_a[1] - vm_pos[1]), merge=False
    ) is True
    assert vm1.terminal_scene_positions_px()["pos"] == r1_a
    _assert_divider_unchanged()
    assert scene.move_components_by([vm1], 20.0, 0.0, merge=False) is True
    _assert_divider_unchanged()

    # Also verify a FreeEnd lying at a component terminal's coordinates does NOT connect.
    r_iso = scene.place_new_component("resistor", QPointF(400.0, 200.0))
    r_iso_a = r_iso.terminal_scene_positions_px()["a"]
    w_free_on_pin = scene.add_wire(TerminalRef("R2", "a"), FreeEnd(*r_iso_a))
    assert w_free_on_pin is not None
    conn_free = scene.extract_connectivity_result()
    assert "R_iso" not in conn_free.component_nodes or r_iso.component.name in conn_free.unconnected_components
    scene.undo_stack.undo()  # remove w_free_on_pin
    scene.undo_stack.undo()  # remove r_iso

    # Delete & Junction merge: deleting w_gnd collapses degree-3 junction into degree-2 hint bend;
    # moving an attached component clears stale hints cleanly; deleting JunctionItem creates FreeEnds.
    scene.clearSelection()
    w_gnd.setSelected(True)
    assert scene.delete_selected() is True
    assert len(scene.junction_items) == 0
    assert any(w.hints for w in scene.wire_graph.wires)
    # Move R2 -> incident wire clears stale hint and routes cleanly
    assert scene.move_components_by([r2], 20.0, 0.0, merge=False) is True
    r2_wires = [
        w
        for w in scene.wire_graph.wires
        if (isinstance(w.end_a, TerminalRef) and w.end_a.component == "R2")
        or (isinstance(w.end_b, TerminalRef) and w.end_b.component == "R2")
    ]
    assert all(not w.hints for w in r2_wires)
    scene.undo_stack.undo()  # undo move R2 (restores old hint)
    scene.undo_stack.undo()  # undo delete w_gnd (restores junction)
    assert len(scene.junction_items) == 1

    # Deleting the JunctionItem directly converts its 3 incident wires to FreeEnd endpoints
    scene.clearSelection()
    jitem = scene.junction_items[0]
    jitem.setSelected(True)
    assert scene.delete_selected() is True
    assert len(scene.junction_items) == 0
    assert len(scene.wire_graph.junctions) == 0
    assert any(
        isinstance(w.end_a, FreeEnd) or isinstance(w.end_b, FreeEnd)
        for w in scene.wire_graph.wires
    )
    scene.undo_stack.undo()
    assert len(scene.junction_items) == 1
    _assert_divider_unchanged()

    # Deleting R1 deletes its attached wires and undo restores everything identically.
    scene.clearSelection()
    r1.setSelected(True)
    assert scene.delete_selected() is True
    assert all(it.component.name != "R1" for it in scene.placed_items)
    assert not scene.wire_graph.has_wire(w_top.wire_id)
    scene.undo_stack.undo()
    assert any(it.component.name == "R1" for it in scene.placed_items)
    assert scene.wire_graph.has_wire(w_top.wire_id)
    _assert_divider_unchanged()


def test_phase4_4_random_100_ops_fuzz_with_undo_redo(qapp) -> None:
    """Seeded 100 random moves/rotations/flips keep WireGraph invariants, net partition,
    and hand-calculated node voltages identical after every step, with full undo/redo.
    """
    scene = CircuitScene(SymbolLibrary())
    v1 = scene.place_new_component("dc_voltage", QPointF(-200.0, 0.0), rotation_deg=90)
    r1 = scene.place_new_component("resistor", QPointF(-40.0, -40.0))
    r2 = scene.place_new_component("resistor", QPointF(120.0, 0.0), rotation_deg=90)
    r3 = scene.place_new_component("resistor", QPointF(280.0, 0.0), rotation_deg=90)
    vm1 = scene.place_new_component("voltmeter", QPointF(440.0, 0.0), rotation_deg=90)
    gnd = scene.place_new_component("ground", QPointF(-200.0, 160.0))

    scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
    scene.edit_component_parameters(r1, {"resistance": 1000.0})
    scene.edit_component_parameters(r2, {"resistance": 2000.0})
    scene.edit_component_parameters(r3, {"resistance": 2000.0})
    scene.edit_component_parameters(vm1, {"is_ideal": True})

    w1 = scene.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    w2 = scene.add_wire(TerminalRef("R1", "b"), TerminalRef("R2", "a"))
    w3 = scene.add_wire([_interior_grid_point(w2.points), r3.terminal_scene_positions_px()["a"]])
    w4 = scene.add_wire([_interior_grid_point(w3.points), vm1.terminal_scene_positions_px()["pos"]])
    w5 = scene.add_wire(TerminalRef("V1", "neg"), TerminalRef("R2", "b"))
    w6 = scene.add_wire([_interior_grid_point(w5.points), gnd.terminal_scene_positions_px()["gnd"]])
    w7 = scene.add_wire([_interior_grid_point(w6.points), r3.terminal_scene_positions_px()["b"]])
    w8 = scene.add_wire([_interior_grid_point(w7.points), vm1.terminal_scene_positions_px()["neg"]])
    assert all(w is not None for w in (w1, w2, w3, w4, w5, w6, w7, w8))

    assert len(scene.placed_items) == 6
    assert len(scene.wire_graph.wires) >= 8

    # Hand calculation: R_eq = R2 || R3 = (2k * 2k) / (2k + 2k) = 1000 ohm
    # V_mid = 12.0 * 1000.0 / (1000.0 + 1000.0) = 6.0 V
    expected_v_mid = 12.0 * 1000.0 / (1000.0 + 1000.0)
    parts = [v1, r1, r2, r3, vm1, gnd]
    expected_terminals = {
        it.component.name: tuple(it.terminal_scene_positions_px()) for it in parts
    }
    initial_partition = _net_partition(scene.extract_connectivity_result())

    def _state_snapshot() -> tuple[tuple[tuple[str, float, float, int, bool], ...], object]:
        comp_state = tuple(
            (it.component.name, it.pos().x(), it.pos().y(), it.rotation_deg, it.flipped_h)
            for it in parts
        )
        return (comp_state, scene.wire_graph)

    initial_snapshot = _state_snapshot()
    undo_start_idx = scene.undo_stack.index()

    rng = random.Random(20260404)
    offsets = (-60.0, -40.0, -20.0, 20.0, 40.0, 60.0)
    for _ in range(100):
        k = rng.choice((1, 2))
        chosen = rng.sample(parts, k)
        scene.clearSelection()
        for it in chosen:
            it.setSelected(True)
        op = rng.choice(("move", "move", "rotate", "flip"))
        if op == "move":
            dx = rng.choice(offsets)
            dy = rng.choice(offsets)
            assert scene.move_components_by(chosen, dx, dy, merge=False) is True
        elif op == "rotate":
            assert scene.rotate_selected() is True
        else:
            assert scene.flip_selected() is True

        assert scene.wire_graph.problems(expected_terminals) == []
        conn = scene.extract_connectivity_result()
        assert _net_partition(conn) == initial_partition
        sol = MNASolver(conn.circuit).solve_dc()
        mid_node = conn.component_nodes["R2"]["a"]
        assert math.isclose(sol.node_voltages[mid_node], expected_v_mid, rel_tol=1e-6)
        assert math.isclose(conn.circuit.get_component("VM1").reading, expected_v_mid, rel_tol=1e-6)

    final_snapshot = _state_snapshot()
    assert scene.undo_stack.index() == undo_start_idx + 100

    for _ in range(100):
        scene.undo_stack.undo()
    assert _state_snapshot() == initial_snapshot

    for _ in range(100):
        scene.undo_stack.redo()
    assert _state_snapshot() == final_snapshot

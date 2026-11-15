"""Tests for Phase 4 (`Step 4A-1`) integration of `circuitsim.wiring` into `CircuitScene`."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF

from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine import MNASolver
from circuitsim.wiring.model import FreeEnd, JunctionRef, TerminalRef


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

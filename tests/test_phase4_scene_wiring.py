"""Tests for Phase 4 (`Step 4A-1`) integration of `circuitsim.wiring` into `CircuitScene`."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter

from circuitsim.app.themes import ThemeMode, get_theme_palette
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine import MNASolver
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

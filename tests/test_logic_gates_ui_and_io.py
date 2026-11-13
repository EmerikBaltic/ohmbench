"""UI, symbol geometry, orientation, wiring v2, QuickValueDialog, and .circ I/O tests for Logic Gates (Chat 2)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from circuitsim.app.palette_dock import (
    CATEGORY_ORDER,
    COMPONENT_ORDER_IN_CATEGORY,
    ComponentPaletteWidget,
)
from circuitsim.app.themes import ThemeMode
from circuitsim.canvas.component_item import (
    ComponentItem,
    format_component_value_summary,
)
from circuitsim.canvas.connectivity import (
    PlacedComponentSpec,
    WireSpec,
    extract_connectivity,
)
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_geometry import (
    DEFAULT_IEC_SYMBOLS_DIR,
    GRID_PITCH_PX,
    load_all_iec_metadata,
    logic_gate_marks,
    prepare_svg_for_theme,
    transform_terminal_position,
)
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.components import (
    DCVoltageSource,
    Ground,
    LOGIC_FAMILY_PRESETS,
    LogicGate,
    Resistor,
    Switch,
    create_default_component_model,
)
from circuitsim.components.logic_gate import GATE_COMPONENT_TYPES
from circuitsim.dialogs.quick_edit_dialog import QuickValueDialog
from circuitsim.engine import MNASolver, SimulationSession, SimulationState
from circuitsim.io.circ_serializer import (
    CircuitDocument,
    ViewState,
    dumps_document,
    loads_document,
)
from circuitsim.wiring import (
    CircuitDocumentV2,
    TerminalRef,
    Wire,
    WireGraph,
    build_circuit,
    dumps_document_v2,
    loads_document_v2,
    migrate_v1_to_v2,
)


def _render_component_to_rgba(
    item: ComponentItem,
    canvas_size: int = 240,
) -> tuple[np.ndarray, int]:
    """Render *item* onto a transparent RGBA QImage centered at `(canvas_size//2, canvas_size//2)`."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QStyleOptionGraphicsItem

    half = canvas_size // 2
    img = QImage(canvas_size, canvas_size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.translate(QPointF(float(half), float(half)))
    item.paint(painter, QStyleOptionGraphicsItem(), None)
    painter.end()

    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes()).reshape(
        canvas_size, canvas_size, 4
    ).copy()
    return arr, half


def test_logic_gates_metadata_svgs_and_palette(qapp) -> None:
    """Verify palette registration, extension metadata loading, grid-aligned pins, and SVG validity."""
    assert "Logic Gates" in CATEGORY_ORDER
    assert COMPONENT_ORDER_IN_CATEGORY["Logic Gates"] == GATE_COMPONENT_TYPES

    core_catalog = load_all_iec_metadata(
        DEFAULT_IEC_SYMBOLS_DIR, include_extensions=False
    )
    assert len(core_catalog) == 11
    for gid in GATE_COMPONENT_TYPES:
        assert gid not in core_catalog

    full_catalog = load_all_iec_metadata(
        DEFAULT_IEC_SYMBOLS_DIR, include_extensions=True
    )
    for gid in GATE_COMPONENT_TYPES:
        assert gid in full_catalog
        meta = full_catalog[gid]
        assert meta.category == "Logic Gates"
        expected_grid = (4, 2) if gid == "gate_not" else (4, 4)
        expected_terms = ["in", "out"] if gid == "gate_not" else ["a", "b", "out"]
        assert meta.grid_size == expected_grid
        assert [t.name for t in meta.terminals] == expected_terms

        for term in meta.terminals:
            vx, vy = meta.terminal_in_svg_viewbox(term)
            assert vx % GRID_PITCH_PX == 0
            assert vy % GRID_PITCH_PX == 0

        svg_path = meta.svg_path_for_state(None)
        assert svg_path.exists()
        raw_svg = svg_path.read_text(encoding="utf-8")
        root = ET.fromstring(raw_svg)
        text_nodes = [
            el for el in root.iter() if el.tag.lower().endswith("}text") or el.tag.lower() == "text"
        ]
        assert not text_nodes, f"{gid}.svg must use vector <path> qualifiers, not <text>."

        for stroke in ("#E6EDF3", "#1B1F24"):
            prep = prepare_svg_for_theme(svg_path, meta, stroke_color=stroke)
            assert prep.is_valid, f"{gid}.svg failed SVG prep: {prep.warning}"

    lib = SymbolLibrary()
    dock = ComponentPaletteWidget(lib)
    tree = dock.tree
    found_categories = [
        tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())
    ]
    assert "Logic Gates" in found_categories
    logic_idx = found_categories.index("Logic Gates")
    logic_cat_item = tree.topLevelItem(logic_idx)
    assert logic_cat_item.childCount() == len(GATE_COMPONENT_TYPES)
    for i, gid in enumerate(GATE_COMPONENT_TYPES):
        child = logic_cat_item.child(i)
        assert child.data(0, 0x0100) == gid  # Qt.ItemDataRole.UserRole == 0x0100
        assert not child.icon(0).isNull()

    dock.search_input.setText("NAND")
    assert not logic_cat_item.isHidden()
    hidden_flags = {
        logic_cat_item.child(i).data(0, 0x0100): logic_cat_item.child(i).isHidden()
        for i in range(logic_cat_item.childCount())
    }
    assert hidden_flags["gate_nand"] is False
    assert hidden_flags["gate_and"] is True


@pytest.mark.parametrize("rotation_deg", [0, 90, 180, 270])
@pytest.mark.parametrize("flipped_h", [False, True])
def test_logic_gate_orientations_terminals_and_upright_qualifier(
    qapp, rotation_deg: int, flipped_h: bool
) -> None:
    """Verify hand-calculated pin positions, paint bounds, and upright '1' qualifier in all 8 orientations."""
    # Hand-calculated binary gate pins for a=(-2,-1), b=(-2,1), out=(2,0) with grid=20px around (0, 0):
    expected_and: dict[tuple[int, bool], dict[str, tuple[float, float]]] = {
        (0, False): {"a": (-40.0, -20.0), "b": (-40.0, 20.0), "out": (40.0, 0.0)},
        (0, True): {"a": (40.0, -20.0), "b": (40.0, 20.0), "out": (-40.0, 0.0)},
        (90, False): {"a": (20.0, -40.0), "b": (-20.0, -40.0), "out": (0.0, 40.0)},
        (90, True): {"a": (-20.0, -40.0), "b": (20.0, -40.0), "out": (0.0, 40.0)},
        (180, False): {"a": (40.0, 20.0), "b": (40.0, -20.0), "out": (-40.0, 0.0)},
        (180, True): {"a": (-40.0, 20.0), "b": (-40.0, -20.0), "out": (40.0, 0.0)},
        (270, False): {"a": (-20.0, 40.0), "b": (20.0, 40.0), "out": (0.0, -40.0)},
        (270, True): {"a": (20.0, 40.0), "b": (-20.0, 40.0), "out": (0.0, -40.0)},
    }
    and_marks = logic_gate_marks(
        "gate_and", (0.0, 0.0), rotation_deg=rotation_deg, flipped_h=flipped_h
    )
    assert dict(and_marks["terminals"]) == expected_and[(rotation_deg, flipped_h)]

    lib = SymbolLibrary()
    # Check uprightness of NOT gate's asymmetric '1' qualifier against reference (0, False).
    ref_gate = create_default_component_model("gate_not", "Uref")
    ref_item = ComponentItem(
        ref_gate,
        lib,
        rotation_deg=0,
        flipped_h=False,
        theme=ThemeMode.DARK,
    )
    ref_rgba, ref_half = _render_component_to_rgba(ref_item, canvas_size=220)
    ref_bx, ref_by = logic_gate_marks(
        "gate_not", (0.0, 0.0), rotation_deg=0, flipped_h=False
    )["body_center"]
    rx0, ry0 = int(round(ref_half + ref_bx)), int(round(ref_half + ref_by))
    ref_mask = ref_rgba[ry0 - 8 : ry0 + 8, rx0 - 8 : rx0 + 8, 3] > 32
    assert np.count_nonzero(ref_mask) > 10

    for gid in GATE_COMPONENT_TYPES:
        for theme in (ThemeMode.DARK, ThemeMode.LIGHT):
            comp = create_default_component_model(gid, "U1")
            comp.family = "5V_CMOS"
            item = ComponentItem(
                comp,
                lib,
                rotation_deg=rotation_deg,
                flipped_h=flipped_h,
                theme=theme,
            )
            item.setSelected(True)
            item._is_hovered = True
            if theme == ThemeMode.DARK and gid == "gate_nand":
                item.set_diagnostic("warning", "Gate check")
            rgba, half = _render_component_to_rgba(item, canvas_size=260)
            ys, xs = np.nonzero(rgba[:, :, 3] > 0)
            assert len(xs) > 0
            brect = item.boundingRect()
            # Allow 2 px antialiasing feather around boundingRect.
            min_x = float(xs.min() - half)
            max_x = float(xs.max() - half)
            min_y = float(ys.min() - half)
            max_y = float(ys.max() - half)
            assert min_x >= brect.left() - 2.0
            assert max_x <= brect.right() + 2.0
            assert min_y >= brect.top() - 2.0
            assert max_y <= brect.bottom() + 2.0

            if gid == "gate_not" and theme == ThemeMode.DARK:
                marks = logic_gate_marks(
                    "gate_not", (0.0, 0.0), rotation_deg=rotation_deg, flipped_h=flipped_h
                )
                bx, by = marks["body_center"]
                cx, cy = int(round(half + bx)), int(round(half + by))
                glyph_mask = rgba[cy - 8 : cy + 8, cx - 8 : cx + 8, 3] > 32
                iou = np.count_nonzero(glyph_mask & ref_mask) / float(
                    np.count_nonzero(glyph_mask | ref_mask)
                )
                assert iou > 0.70, (
                    f"'1' qualifier inverted or mirrored at ({rotation_deg}, {flipped_h}), IoU={iou:.2f}"
                )


def test_logic_gate_scene_wiring_connectivity_and_live_simulation(qapp) -> None:
    """Place, wire, name ('U1', 'U2'), extract connectivity, and simulate AND + NOT gates end-to-end."""
    scene = CircuitScene(SymbolLibrary())
    vcc = scene.place_new_component("dc_voltage", (0.0, 0.0), rotation_deg=180)
    vcc.component.set_parameter_value("dc_voltage", 5.0)
    gnd_v = scene.place_new_component("ground", (-40.0, 60.0))
    sw = scene.place_new_component("switch", (100.0, 0.0))
    g_and = scene.place_new_component("gate_and", (220.0, 20.0))
    g_not = scene.place_new_component("gate_not", (340.0, 20.0))
    r_pulldown = scene.place_new_component("resistor", (140.0, 80.0))
    r_pulldown.component.set_parameter_value("resistance", 10_000.0)
    gnd_r = scene.place_new_component("ground", (180.0, 140.0))

    assert g_and.component.name == "U1"
    assert g_not.component.name == "U2"
    assert format_component_value_summary(g_and.component) == "5V CMOS"

    v_pos = vcc.terminal_scene_positions()["pos"]
    v_neg = vcc.terminal_scene_positions()["neg"]
    g1_pin = gnd_v.terminal_scene_positions()["gnd"]
    sw_a = sw.terminal_scene_positions()["a"]
    sw_b = sw.terminal_scene_positions()["b"]
    and_a = g_and.terminal_scene_positions()["a"]
    and_b = g_and.terminal_scene_positions()["b"]
    and_out = g_and.terminal_scene_positions()["out"]
    not_in = g_not.terminal_scene_positions()["in"]
    rp_a = r_pulldown.terminal_scene_positions()["a"]
    rp_b = r_pulldown.terminal_scene_positions()["b"]
    g2_pin = gnd_r.terminal_scene_positions()["gnd"]

    for p1, p2 in (
        (v_neg, g1_pin),
        (v_pos, sw_a),
        (sw_b, and_a),
        (and_a, and_b),
        (and_b, rp_a),
        (rp_b, g2_pin),
        (and_out, not_in),
    ):
        assert scene.add_routed_wire(p1, p2) is not None

    conn = scene.extract_connectivity_result()
    assert not conn.unconnected_components
    circuit = conn.circuit
    u1_model = circuit.get_component("U1")
    u2_model = circuit.get_component("U2")
    assert isinstance(u1_model, LogicGate)
    assert isinstance(u2_model, LogicGate)
    # Regression check for Step 1: input_nodes/output_node track dynamically rebound .nodes
    assert u1_model.input_nodes == u1_model.nodes[:-1]
    assert u1_model.output_node == u1_model.nodes[-1]
    assert u1_model.output_node == u2_model.input_nodes[0]

    session = SimulationSession(circuit, connectivity=conn)
    # Run with switch open -> AND inputs ~0V -> AND out = L (~0.05V), NOT out = H (~4.95V)
    session.start()
    frame_open = session.step_chunk(steps=15)
    assert frame_open.state == SimulationState.RUNNING
    assert frame_open.components["U1"].logic_level == "L"
    assert frame_open.components["U2"].logic_level == "H"
    assert math.isclose(
        frame_open.node_voltages[u2_model.output_node], 4.95, abs_tol=0.1
    )

    scene.apply_simulation_frame(frame_open)
    assert g_not.live_display_value is not None
    assert "H" in g_not.live_display_value
    assert "Output Level: H" in g_not.get_live_tooltip(is_running=True)

    # Close switch -> inputs rise to 5V -> AND out = H (~4.95V), NOT out = L (~0.05V)
    session.toggle_switch(sw.component.name)
    frame_closed = session.step_chunk(steps=15)
    assert frame_closed.components["U1"].logic_level == "H"
    assert frame_closed.components["U2"].logic_level == "L"
    assert math.isclose(
        frame_closed.node_voltages[u2_model.output_node], 0.05, abs_tol=0.1
    )

    # Atomic live batch parameter update while RUNNING (shifts U2 rails cleanly)
    frame_edited = session.update_component_parameter(
        "U2",
        "__batch__",
        {"v_il": 0.6, "v_ih": 1.8, "v_ol": 0.2, "v_oh": 3.3, "family": "Custom"},
    )
    frame_settled = session.step_chunk(steps=15)
    assert frame_edited.components["U2"].logic_level in ("H", "L")
    assert frame_settled.components["U2"].logic_level == "L"
    assert math.isclose(
        frame_settled.node_voltages[u2_model.output_node], 0.2, abs_tol=0.1
    )


def test_logic_gates_wiring_v2_netlist_and_migration() -> None:
    """Verify v2 WireGraph/build_circuit and v1->v2 migration support extension logic gate symbols."""
    vcc = DCVoltageSource("V1", "n1", "0", voltage=5.0)
    gnd = Ground("GND1", "0")
    nand_gate = LogicGate("U1", "nand", ("in_a", "in_b"), "mid")
    not_gate = LogicGate("U2", "not", "mid", "out")

    comps = [vcc, gnd, nand_gate, not_gate]
    graph = WireGraph()
    graph = graph.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
    graph = graph.add_wire(TerminalRef("V1", "pos"), TerminalRef("U1", "a"))
    graph = graph.add_wire(TerminalRef("U1", "a"), TerminalRef("U1", "b"))
    graph = graph.add_wire(TerminalRef("U1", "out"), TerminalRef("U2", "in"))

    # build_circuit with default catalog (must include extension symbols automatically!)
    net = build_circuit(graph, comps)
    solver = MNASolver(net.circuit)
    # NAND(1, 1) = 0 (~0.05 V), NOT(0) = 1 (~4.95 V)
    for _step in range(40):
        res = solver.step_transient(5e-9)
    u2_out_node = net.circuit.get_component("U2").output_node
    assert math.isclose(res.node_voltages[u2_out_node], 4.95, abs_tol=0.1)

    # Test migrate_v1_to_v2 on a v1 document containing XOR/XNOR gates
    u_xor = LogicGate("U3", "xor", ("x1", "x2"), "x3")
    spec_v1 = PlacedComponentSpec(
        component=u_xor, center_px=(100.0, 100.0), rotation_deg=0, flipped_h=False
    )
    meta_catalog = load_all_iec_metadata(DEFAULT_IEC_SYMBOLS_DIR, include_extensions=True)
    pins = spec_v1.resolve_terminals_px(meta_catalog)
    wires_v1 = [
        WireSpec("W1", ((0.0, 80.0), pins["a"])),
        WireSpec("W2", ((0.0, 120.0), pins["b"])),
        WireSpec("W3", (pins["out"], (200.0, 100.0))),
    ]
    v1_json = dumps_document(CircuitDocument(placed=[spec_v1], wires=wires_v1))
    v2_doc = migrate_v1_to_v2(v1_json)
    term_refs = [
        end
        for w in v2_doc.graph.wires
        for end in (w.end_a, w.end_b)
        if isinstance(end, TerminalRef)
    ]
    assert TerminalRef("U3", "a") in term_refs
    assert TerminalRef("U3", "b") in term_refs
    assert TerminalRef("U3", "out") in term_refs


def test_quick_edit_dialog_logic_gate_presets_custom_validation_and_undo_redo(qapp) -> None:
    """Verify QuickValueDialog preset sync, Custom switch, invalid threshold guard, and batch Undo/Redo."""
    scene = CircuitScene(SymbolLibrary())
    item = scene.place_new_component("gate_and", (100.0, 100.0))
    gate = item.component
    assert isinstance(gate, LogicGate)

    dlg = QuickValueDialog(gate)
    assert dlg.gate_family_combo is not None
    assert dlg.gate_initial_combo is not None
    assert dlg.gate_family_combo.currentData() == "5V_CMOS"
    assert dlg.gate_initial_combo.currentData() is False

    # Switch preset -> 5V TTL auto-fills all 4 thresholds/rails
    ttl_idx = dlg.gate_family_combo.findData("5V_TTL")
    assert ttl_idx >= 0
    dlg.gate_family_combo.setCurrentIndex(ttl_idx)
    ttl = LOGIC_FAMILY_PRESETS["5V_TTL"]
    assert float(dlg.inputs["v_il"].text()) == ttl["v_il"]
    assert float(dlg.inputs["v_ih"].text()) == ttl["v_ih"]
    assert float(dlg.inputs["v_ol"].text()) == ttl["v_ol"]
    assert float(dlg.inputs["v_oh"].text()) == ttl["v_oh"]

    # Manual edit of a threshold switches preset combo cleanly to Custom
    dlg.inputs["v_ih"].setText("2.6")
    assert dlg.gate_family_combo.currentData() == "Custom"

    # Inverted thresholds must fail validation cleanly with an error banner
    dlg.inputs["v_il"].setText("3.0")
    dlg.inputs["v_ih"].setText("2.0")
    parsed, err = dlg.validate_inputs()
    assert parsed is None
    assert err is not None and "V_IL" in err
    assert dlg._apply_current_inputs() is False
    assert not dlg.error_label.isHidden()
    assert "V_IL" in dlg.error_label.text()

    # Cross-range threshold shift (1.5..3.5 -> 0.4..1.1) + Initial Output High + Undo/Redo
    dlg.inputs["v_il"].setText("0.4")
    dlg.inputs["v_ih"].setText("1.1")
    high_idx = dlg.gate_initial_combo.findData(True)
    dlg.gate_initial_combo.setCurrentIndex(high_idx)
    parsed_ok, err_ok = dlg.validate_inputs()
    assert err_ok is None and parsed_ok is not None
    assert scene.edit_component_parameters(item, parsed_ok, show_value=True)

    assert gate.family == "Custom"
    assert math.isclose(gate.v_il, 0.4)
    assert math.isclose(gate.v_ih, 1.1)
    assert gate.initial_state is True
    assert gate.output_level is True

    # Cross-range shift upward (0.4..1.1 -> 3.6..4.3) works atomically without ordering failure
    assert scene.edit_component_parameters(
        item, {"v_il": 3.6, "v_ih": 4.3, "family": "Custom"}
    )
    assert math.isclose(gate.v_il, 3.6)
    assert math.isclose(gate.v_ih, 4.3)

    # Undo restores (0.4, 1.1), second undo restores 5V_CMOS (1.5, 3.5) and initial_state=False
    scene.undo_stack.undo()
    assert math.isclose(gate.v_il, 0.4)
    assert math.isclose(gate.v_ih, 1.1)
    scene.undo_stack.undo()
    assert gate.family == "5V_CMOS"
    assert math.isclose(gate.v_il, 1.5)
    assert math.isclose(gate.v_ih, 3.5)
    assert gate.initial_state is False
    assert gate.output_level is False

    # Redo restores Custom (0.4, 1.1) and initial_state=True
    scene.undo_stack.redo()
    assert gate.family == "Custom"
    assert math.isclose(gate.v_il, 0.4)
    assert gate.initial_state is True


def test_logic_gates_circ_v1_and_v2_round_trip() -> None:
    """Save and reload logic gates with custom thresholds, family, initial_state, and orientation in v1 & v2."""
    gate_ttl = LogicGate("U1", "nand", ("in_a", "in_b"), "mid", t_pd=25e-9)
    gate_ttl.set_family_preset("5V_TTL")
    gate_custom = LogicGate(
        "U2",
        "not",
        "mid",
        "out",
        v_il=0.7,
        v_ih=2.4,
        v_ol=0.1,
        v_oh=4.8,
        initial_state=True,
    )
    vcc = DCVoltageSource("V1", "in_a", "0", voltage=5.0)
    gnd = Ground("GND1", "0")

    placed = [
        PlacedComponentSpec(vcc, (0.0, 0.0), rotation_deg=0, flipped_h=False),
        PlacedComponentSpec(gnd, (-40.0, 40.0), rotation_deg=0, flipped_h=False),
        PlacedComponentSpec(gate_ttl, (160.0, 40.0), rotation_deg=90, flipped_h=True),
        PlacedComponentSpec(gate_custom, (280.0, 40.0), rotation_deg=180, flipped_h=False),
    ]
    catalog = load_all_iec_metadata(DEFAULT_IEC_SYMBOLS_DIR, include_extensions=True)
    v1_pins = placed[0].resolve_terminals_px(catalog)
    g_pins = placed[1].resolve_terminals_px(catalog)
    u1_pins = placed[2].resolve_terminals_px(catalog)
    u2_pins = placed[3].resolve_terminals_px(catalog)

    # v1 round-trip
    doc_v1 = CircuitDocument(
        placed=placed,
        wires=[
            WireSpec("W1", (v1_pins["neg"], g_pins["gnd"])),
            WireSpec("W2", (v1_pins["pos"], u1_pins["a"])),
            WireSpec("W3", (u1_pins["a"], u1_pins["b"])),
            WireSpec("W4", (u1_pins["out"], u2_pins["in"])),
        ],
        view=ViewState(theme="light", zoom=1.25),
    )
    loaded_v1 = loads_document(dumps_document(doc_v1))
    u1_loaded = next(p for p in loaded_v1.placed if p.component.name == "U1")
    u2_loaded = next(p for p in loaded_v1.placed if p.component.name == "U2")
    assert isinstance(u1_loaded.component, LogicGate)
    assert isinstance(u2_loaded.component, LogicGate)
    assert u1_loaded.rotation_deg == 90 and u1_loaded.flipped_h is True
    assert u1_loaded.component.family == "5V_TTL"
    assert math.isclose(u1_loaded.component.t_pd, 25e-9)
    assert u2_loaded.rotation_deg == 180 and u2_loaded.flipped_h is False
    assert u2_loaded.component.family == "Custom"
    assert math.isclose(u2_loaded.component.v_il, 0.7)
    assert math.isclose(u2_loaded.component.v_ih, 2.4)
    assert u2_loaded.component.initial_state is True
    assert u2_loaded.component.output_level is True

    # v2 round-trip and MNA solve
    graph = WireGraph()
    graph = graph.add_wire(TerminalRef("V1", "neg"), TerminalRef("GND1", "gnd"))
    graph = graph.add_wire(TerminalRef("V1", "pos"), TerminalRef("U1", "a"))
    graph = graph.add_wire(TerminalRef("U1", "a"), TerminalRef("U1", "b"))
    graph = graph.add_wire(TerminalRef("U1", "out"), TerminalRef("U2", "in"))

    doc_v2 = CircuitDocumentV2(
        placed=placed,
        graph=graph,
        view=ViewState(theme="dark", zoom=1.0),
    )
    loaded_v2 = loads_document_v2(dumps_document_v2(doc_v2))
    net = build_circuit(loaded_v2.graph, [p.component for p in loaded_v2.placed])
    solver = MNASolver(net.circuit)
    # U1 is 5V_TTL NAND(5V, 5V) -> V_OL=0.4 V -> below U2 V_IL=0.7 V -> U2 NOT outputs V_OH=4.8 V
    for _step in range(40):
        res = solver.step_transient(5e-9)
    u2_out = net.circuit.get_component("U2").output_node
    assert math.isclose(res.node_voltages[u2_out], 4.8, abs_tol=0.1)

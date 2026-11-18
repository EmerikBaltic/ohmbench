"""Phase H Workbench Expansion tests (H0-H10) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.app.palette_dock import ComponentPaletteWidget
from circuitsim.app.themes import ThemeMode, build_stylesheet, get_theme_palette
from circuitsim.canvas.component_item import ComponentItem
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.components import (
    DCVoltageSource,
    Ground,
    KNOWN_COMPONENT_TYPES,
    NPNBJT,
    PNPBJT,
    Resistor,
    ZenerDiode,
    create_default_component_model,
)
from circuitsim.engine import Circuit, MNASolver, evaluate_component_stress
from circuitsim.io.spice_import import (
    instantiate_bjt_model,
    instantiate_diode_model,
    parse_spice_library,
)
from circuitsim.wiring.model import TerminalRef
from circuitsim.wiring.serializer import dumps_document_v2, loads_document_v2


def test_h0_phase_h_scaffolding(qapp) -> None:
    """Verify offscreen Qt scaffolding and empty CircuitScene setup."""
    assert qapp is not None
    scene = CircuitScene(SymbolLibrary())
    try:
        assert scene.placed_items == []
        assert scene.placed_wires == []
        assert scene.wire_graph.wires == ()
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_h1_parallel_voltmeter_inductor_straight_2point_wires(qapp) -> None:
    """H1: Stacked horizontal VM1 above L1 in parallel connects with straight 2-point vertical wires."""
    scene = CircuitScene(SymbolLibrary())
    try:
        l1 = scene.place_new_component("inductor", QPointF(0.0, 0.0))
        vm1 = scene.place_new_component("voltmeter", QPointF(0.0, -120.0))

        l1_terms = l1.terminal_scene_positions_px()
        vm1_terms = vm1.terminal_scene_positions_px()
        x_left = l1_terms["a"][0]
        x_right = l1_terms["b"][0]
        assert vm1_terms["pos"][0] == x_left
        assert vm1_terms["neg"][0] == x_right

        w_left = scene.add_wire(TerminalRef("VM1", "pos"), TerminalRef("L1", "a"))
        w_right = scene.add_wire(TerminalRef("VM1", "neg"), TerminalRef("L1", "b"))
        assert w_left is not None
        assert w_right is not None
        assert len(w_left.points) == 2
        assert len(w_right.points) == 2
        assert all(px == x_left for px, _py in w_left.points)
        assert all(px == x_right for px, _py in w_right.points)
        assert w_left.points == (vm1_terms["pos"], l1_terms["a"])
        assert w_right.points == (vm1_terms["neg"], l1_terms["b"])
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_h2_palette_vertical_scrollbar_search_and_expand_collapse(qapp) -> None:
    """H2: Component palette has visible vertical scrollbar, live search filter, and Expand/Collapse All."""
    palette = ComponentPaletteWidget(SymbolLibrary(), theme=ThemeMode.DARK)
    try:
        # Verify themes.py QSS includes explicit vertical scrollbar rules in both Dark & Light themes
        for mode in (ThemeMode.DARK, ThemeMode.LIGHT):
            qss = build_stylesheet(get_theme_palette(mode))
            assert "QScrollBar:vertical" in qss
            assert "QScrollBar::handle:vertical" in qss

        palette.resize(240, 300)
        palette.show()
        palette.expand_all()
        qapp.processEvents()

        vbar = palette.tree.verticalScrollBar()
        assert palette.tree.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
        assert vbar.maximum() > 0
        assert vbar.isVisible()

        # Collapse All then filter for "mcu" -> Mixed-Signal auto-expands, others hidden
        palette.btn_collapse_all.click()
        qapp.processEvents()
        palette.search_input.setText("mcu")
        qapp.processEvents()

        visible_cats: list[str] = []
        mcu_visible = False
        for i in range(palette.tree.topLevelItemCount()):
            cat = palette.tree.topLevelItem(i)
            assert cat is not None
            if not cat.isHidden():
                visible_cats.append(cat.text(0))
                assert cat.isExpanded()
                for j in range(cat.childCount()):
                    ch = cat.child(j)
                    if ch is not None and not ch.isHidden() and ch.text(0) == "MCU Block":
                        mcu_visible = True
        assert mcu_visible
        assert visible_cats == ["Mixed-Signal"]

        # Clearing search restores all categories; Expand All works cleanly
        palette.search_input.clear()
        palette.btn_expand_all.click()
        qapp.processEvents()
        hidden_cats = [
            palette.tree.topLevelItem(i).text(0)
            for i in range(palette.tree.topLevelItemCount())
            if palette.tree.topLevelItem(i).isHidden()
        ]
        assert hidden_cats == []
    finally:
        palette.close()
        qapp.processEvents()


def test_h3_zener_regulator_npn_pnp_emitter_follower_and_spice(qapp) -> None:
    """H3: Zener 5.1V regulator, NPN/PNP emitter followers, pinouts, stress, and SPICE import."""
    for ctype in ("zener_diode", "npn_bjt", "pnp_bjt"):
        assert ctype in KNOWN_COMPONENT_TYPES
        comp = create_default_component_model(ctype, f"test_{ctype}")
        assert comp.get_editable_parameters()

    # (a) Zener Voltage Regulator (unloaded & loaded)
    for r_load in (None, 1000.0):
        c_z = Circuit()
        c_z.add(Ground("GND", "0"))
        c_z.add(DCVoltageSource("V1", "vin", "0", voltage=12.0))
        c_z.add(Resistor("R_ser", "vin", "vout", resistance=500.0))
        c_z.add(ZenerDiode("DZ1", "0", "vout", v_z=5.1))
        if r_load is not None:
            c_z.add(Resistor("R_load", "vout", "0", resistance=r_load))
        res_z = MNASolver(c_z).solve_dc()
        assert math.isclose(res_z.node_voltages["vout"], 5.10, abs_tol=0.08), (
            f"Zener vout={res_z.node_voltages['vout']}"
        )

    # (b) NPN Emitter Follower
    c_npn = Circuit()
    c_npn.add(Ground("GND", "0"))
    c_npn.add(DCVoltageSource("VCC", "vcc", "0", voltage=10.0))
    c_npn.add(DCVoltageSource("VB", "vb", "0", voltage=5.0))
    c_npn.add(Resistor("R_E", "ve", "0", resistance=1000.0))
    q_npn = NPNBJT("Q1", "vb", "vcc", "ve", beta_f=100.0)
    c_npn.add(q_npn)
    res_npn = MNASolver(c_npn).solve_dc()
    v_e = res_npn.node_voltages["ve"]
    i_e = abs(q_npn.i_e)
    i_b = q_npn.i_b
    assert 4.30 <= v_e <= 4.35, f"Unexpected NPN V_E={v_e}"
    assert math.isclose(i_e, v_e / 1000.0, rel_tol=1e-4)
    assert math.isclose(i_b, i_e / 101.0, rel_tol=0.08)

    # (c) PNP Emitter Follower (symmetric high-side): V_EE = 10 V -> R_E = 1k -> E, B=5V, C=0V
    c_pnp = Circuit()
    c_pnp.add(Ground("GND", "0"))
    c_pnp.add(DCVoltageSource("VEE", "vee", "0", voltage=10.0))
    c_pnp.add(DCVoltageSource("VB", "vb", "0", voltage=5.0))
    c_pnp.add(Resistor("R_E", "vee", "ve", resistance=1000.0))
    q_pnp = PNPBJT("Q2", "vb", "0", "ve", beta_f=100.0)
    c_pnp.add(q_pnp)
    res_pnp = MNASolver(c_pnp).solve_dc()
    v_ep = res_pnp.node_voltages["ve"]
    assert 5.65 <= v_ep <= 5.72, f"Unexpected PNP V_E={v_ep}"
    assert q_pnp.i_b < 0.0  # Base current flows out of PNP base
    assert math.isclose(abs(q_pnp.i_b), abs(q_pnp.i_e) / 101.0, rel_tol=0.08)

    # Pinout & F4 stress & Scene placement & .circ v2 round-trip
    pkg_q, pins_q = ComponentItem.package_and_pinout_labels(q_npn)
    assert "TO-92" in pkg_q
    assert set(pins_q.keys()) == {"B", "C", "E"}
    pkg_z, pins_z = ComponentItem.package_and_pinout_labels(ZenerDiode("DZ1", "0", "out"))
    assert pins_z == {"anode": "Anode", "cathode": "Cathode"}
    stress = evaluate_component_stress(c_npn, res_npn)
    assert "Q1" in stress

    scene = CircuitScene(SymbolLibrary())
    try:
        q_item = scene.place_new_component("npn_bjt", QPointF(0.0, 0.0))
        dz_item = scene.place_new_component("zener_diode", QPointF(160.0, 0.0))
        assert q_item.component.name == "Q1"
        assert dz_item.component.name == "DZ1"
        doc_rt = loads_document_v2(dumps_document_v2(scene.to_document_v2()))
        assert {p.component.component_type for p in doc_rt.placed} == {"npn_bjt", "zener_diode"}
    finally:
        scene.shutdown_scene()

    # F8 SPICE .MODEL NPN / PNP / D(BV=5.1) import
    lib = parse_spice_library(
        ".MODEL Q2N3904 NPN(IS=1e-14 BF=150 VAF=100)\n"
        ".MODEL Q2N3906 PNP(IS=1e-14 BF=120)\n"
        ".MODEL D1N4733 D(IS=1e-14 N=1.0 RS=1.0 BV=5.1)\n"
    )
    imp_npn = instantiate_bjt_model(lib.models["Q2N3904"], "Q1", "b", "c", "e")
    imp_pnp = instantiate_bjt_model(lib.models["Q2N3906"], "Q2", "b", "c", "e")
    imp_dz = instantiate_diode_model(lib.models["D1N4733"], "DZ1", "0", "out", as_zener=True)
    assert isinstance(imp_npn, NPNBJT) and math.isclose(imp_npn.beta_f, 150.0)
    assert isinstance(imp_pnp, PNPBJT) and math.isclose(imp_pnp.beta_f, 120.0)
    assert isinstance(imp_dz, ZenerDiode) and math.isclose(imp_dz.v_z, 5.1)

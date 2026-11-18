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
    MCUBlock,
    NMOS,
    NPNBJT,
    OpAmp,
    PMOS,
    PNPBJT,
    Resistor,
    ZenerDiode,
    create_default_component_model,
)
from circuitsim.dialogs.mcu_firmware_dialog import MCUFirmwareDialog
from circuitsim.engine import Circuit, MNASolver, evaluate_component_stress
from circuitsim.io.share_codec import decode_circuit_share_url, encode_circuit_share_url
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


def test_h4_nmos_pmos_saturation_and_opamp_amplifiers(qapp) -> None:
    """H4: NMOS/PMOS Level-1 saturation, Op-Amp inverting/non-inverting/clipping & UI wiring."""
    # 1. NMOS saturation: V_G = 3.5V, V_S = 0V, Vth = 1.5V, kp = 0.02 -> I_D = 40 mA, V_D = 6.0 V
    c_nmos = Circuit()
    c_nmos.add(Ground("GND", "0"))
    c_nmos.add(DCVoltageSource("VDD", "vdd", "0", voltage=10.0))
    c_nmos.add(DCVoltageSource("VG", "g", "0", voltage=3.5))
    c_nmos.add(Resistor("RD", "vdd", "d", resistance=100.0))
    m_nmos = NMOS("M1", "g", "d", "0", v_th=1.5, k_p=0.02, lambda_=0.0, max_power_w=0.5)
    c_nmos.add(m_nmos)
    res_n = MNASolver(c_nmos).solve_dc()
    assert math.isclose(res_n.node_voltages["d"], 6.0, abs_tol=1e-3)
    assert math.isclose(m_nmos.i_d, 0.040, abs_tol=1e-5)

    # 2. PMOS high-side saturation: S=10V, G=6.5V (V_SG=3.5V), RD=100 ohm to ground -> V_D = 4.0 V
    c_pmos = Circuit()
    c_pmos.add(Ground("GND", "0"))
    c_pmos.add(DCVoltageSource("VDD", "vdd", "0", voltage=10.0))
    c_pmos.add(DCVoltageSource("VG", "g", "0", voltage=6.5))
    c_pmos.add(Resistor("RD", "d", "0", resistance=100.0))
    m_pmos = PMOS("M2", "g", "d", "vdd", v_th=-1.5, k_p=0.02, lambda_=0.0, max_power_w=0.5)
    c_pmos.add(m_pmos)
    res_p = MNASolver(c_pmos).solve_dc()
    assert math.isclose(res_p.node_voltages["d"], 4.0, abs_tol=1e-3)
    assert math.isclose(m_pmos.i_d, -0.040, abs_tol=1e-5)

    # 3. Op-Amp inverting amplifier: Vin=1V, Rin=10k, Rf=20k -> Vout=-2.000V, V(-)=~0V
    c_inv = Circuit()
    c_inv.add(Ground("GND", "0"))
    c_inv.add(DCVoltageSource("VIN", "vin", "0", voltage=1.0))
    c_inv.add(Resistor("RIN", "vin", "n_inv", resistance=10_000.0))
    c_inv.add(Resistor("RF", "n_inv", "out", resistance=20_000.0))
    op_inv = OpAmp("U1", "0", "n_inv", "out")
    c_inv.add(op_inv)
    res_inv = MNASolver(c_inv).solve_dc()
    assert abs(res_inv.node_voltages["n_inv"]) < 1e-3
    assert math.isclose(res_inv.node_voltages["out"], -2.0, abs_tol=0.005)

    # 4. Op-Amp non-inverting amplifier: Vin=2V, Rg=10k, Rf=30k (gain=4) -> Vout=+8.000V
    c_ni = Circuit()
    c_ni.add(Ground("GND", "0"))
    c_ni.add(DCVoltageSource("VIN", "vin", "0", voltage=2.0))
    c_ni.add(Resistor("RG", "n_inv", "0", resistance=10_000.0))
    c_ni.add(Resistor("RF", "n_inv", "out", resistance=30_000.0))
    op_ni = OpAmp("U2", "vin", "n_inv", "out")
    c_ni.add(op_ni)
    res_ni = MNASolver(c_ni).solve_dc()
    assert math.isclose(res_ni.node_voltages["out"], 8.0, abs_tol=0.005)

    # 5. Rail clipping at +15V when ideal target (20V) exceeds v_rail_pos
    c_clip = Circuit()
    c_clip.add(Ground("GND", "0"))
    c_clip.add(DCVoltageSource("VIN", "vin", "0", voltage=5.0))
    c_clip.add(Resistor("RG", "n_inv", "0", resistance=10_000.0))
    c_clip.add(Resistor("RF", "n_inv", "out", resistance=30_000.0))
    op_clip = OpAmp("U3", "vin", "n_inv", "out", v_rail_pos=15.0, v_rail_neg=-15.0)
    c_clip.add(op_clip)
    res_clip = MNASolver(c_clip).solve_dc()
    assert 14.8 <= res_clip.node_voltages["out"] <= 15.05

    # 6. Catalog, Pinout (F12), Stress (F4), and .circ v2 persistence
    for ctype in ("nmos", "pmos", "opamp"):
        assert ctype in KNOWN_COMPONENT_TYPES
        m = create_default_component_model(ctype, f"TEST_{ctype}")
        assert len(m.nodes) == 3

    scene = CircuitScene(SymbolLibrary())
    try:
        q1 = scene.place_new_component("nmos", QPointF(0.0, 0.0))
        q2 = scene.place_new_component("pmos", QPointF(160.0, 0.0))
        u1 = scene.place_new_component("opamp", QPointF(320.0, 0.0))
        assert "TO-92" in ComponentItem.package_and_pinout_labels(q1.component)[0]
        assert "TO-92" in ComponentItem.package_and_pinout_labels(q2.component)[0]
        assert "DIP-8" in ComponentItem.package_and_pinout_labels(u1.component)[0]

        m_nmos.set_max_ratings(max_power=0.1)
        stress = evaluate_component_stress(c_nmos, res_n)
        assert "M1" in stress and stress["M1"].utilization_pct >= 200.0 and stress["M1"].level == "overload"

        payload = dumps_document_v2(scene.to_document_v2())
        loaded = loads_document_v2(payload)
        assert [p.component.component_type for p in loaded.placed] == ["nmos", "pmos", "opamp"]
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_h5_mcu_firmware_dialog_editor_serial_monitor_and_share_url(qapp) -> None:
    """H5: MCUFirmwareDialog editor, syntax error guard, live ADC/GPIO/Serial monitor & share URL."""
    scene = CircuitScene(SymbolLibrary())
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-240.0, 0.0))
        v1.component.dc_voltage = 5.0
        gnd = scene.place_new_component("ground", QPointF(-240.0, 140.0))
        r_top = scene.place_new_component("resistor", QPointF(-100.0, -40.0))
        r_top.component.resistance = 10_000.0
        r_bot = scene.place_new_component("resistor", QPointF(-100.0, 80.0))
        r_bot.component.resistance = 10_000.0
        mcu_item = scene.place_new_component("mcu", QPointF(80.0, 0.0))
        r_d0 = scene.place_new_component("resistor", QPointF(260.0, 0.0))
        r_d0.component.resistance = 1_000.0

        scene.add_wire(TerminalRef(v1.component.name, "neg"), TerminalRef(gnd.component.name, "gnd"))
        scene.add_wire(TerminalRef(v1.component.name, "pos"), TerminalRef(r_top.component.name, "a"))
        scene.add_wire(TerminalRef(r_top.component.name, "b"), TerminalRef(r_bot.component.name, "a"))
        scene.add_wire(TerminalRef(r_bot.component.name, "b"), TerminalRef(gnd.component.name, "gnd"))
        scene.add_wire(TerminalRef(r_bot.component.name, "a"), TerminalRef(mcu_item.component.name, "A0"))
        scene.add_wire(TerminalRef(mcu_item.component.name, "D0"), TerminalRef(r_d0.component.name, "a"))
        scene.add_wire(TerminalRef(r_d0.component.name, "b"), TerminalRef(gnd.component.name, "gnd"))

        dlg = MCUFirmwareDialog(mcu_item, scene=scene)
        try:
            # 1. Invalid syntax is caught cleanly by Verify without crashing or closing
            dlg.code_edit.setPlainText("if True print('oops')")
            assert dlg.verify_code() is False
            assert "SyntaxError" in dlg.status_label.text()
            assert dlg.apply_code(close_after=True) is False

            # 2. Valid firmware script compiles & applies
            script = (
                "val = mcu.adc_read('A0')\n"
                "mcu.digital_write('D0', val > 400)\n"
                "mcu.log(f'A0={val}')\n"
            )
            dlg.code_edit.setPlainText(script)
            assert dlg.verify_code() is True
            assert "Compiled OK" in dlg.status_label.text()
            assert dlg.apply_code(close_after=False) is True

            # 3. Solve circuit built from scene -> ADC reads ~512, D0 drives HIGH (>4.5V), Serial logs A0=512
            circ = scene.build_circuit()
            res = MNASolver(circ).solve_dc()
            cloned_mcu = circ.get_component(mcu_item.component.name)
            assert isinstance(cloned_mcu, MCUBlock)
            assert 511 <= cloned_mcu.adc_read("A0") <= 513
            assert res.node_voltages[cloned_mcu.nodes[2]] > 4.5
            assert any("A0=51" in line for line in cloned_mcu.serial_log)

            # Dialog refreshes 6-row live Pin Table & read-only Serial Monitor from canvas MCU mirror
            dlg.refresh_telemetry()
            assert dlg.pin_table.rowCount() == 6
            assert dlg.pin_table.item(0, 0).text() == "A0"
            assert "2.49" in dlg.pin_table.item(0, 2).text() or "2.50" in dlg.pin_table.item(0, 2).text()
            assert "A0=51" in dlg.serial_monitor_edit.toPlainText()
        finally:
            dlg.close()

        # 4. Persistence through .circ v2 and ohmbench://c/ share URL
        share_url = encode_circuit_share_url(scene.to_document_v2())
        decoded_doc = decode_circuit_share_url(share_url)
        decoded_mcus = [p.component for p in decoded_doc.placed if isinstance(p.component, MCUBlock)]
        assert len(decoded_mcus) == 1
        assert "mcu.adc_read('A0')" in decoded_mcus[0].code
    finally:
        scene.shutdown_scene()
        qapp.processEvents()

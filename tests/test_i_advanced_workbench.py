"""Phase I Advanced Workbench tests (I0-I10) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine import SimulationSession
from circuitsim.wiring.model import TerminalRef


def test_i0_phase_i_scaffolding(qapp) -> None:
    """I0: Verify offscreen Qt scaffolding and empty CircuitScene setup."""
    assert qapp is not None
    scene = CircuitScene(SymbolLibrary())
    try:
        assert scene.placed_items == []
        assert scene.placed_wires == []
        assert scene.wire_graph.wires == ()
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_i1_full_span_collision_blocks_overlap_allows_series_and_reverts_drag(qapp) -> None:
    """I1: Full-span footprint blocks overlaps/half-overlaps, allows end-to-end series, and reverts colliding drags."""
    scene = CircuitScene(SymbolLibrary())
    try:
        r1 = scene.place_new_component("resistor", QPointF(0.0, 0.0))
        assert r1 is not None
        assert len(scene.placed_items) == 1

        # All 5 overlapping/half-overlapping positions must be rejected without spawning components
        for bad_pt in (
            QPointF(0.0, 0.0),
            QPointF(20.0, 0.0),
            QPointF(40.0, 0.0),
            QPointF(-40.0, 0.0),
            QPointF(0.0, 20.0),
        ):
            assert scene.place_new_component("resistor", bad_pt) is None
            assert len(scene.placed_items) == 1
            assert "Cannot place component" in scene.last_status_notice

        # End-to-end series placement at (80, 0) touches terminal cleanly at (40, 0) without collision
        r2 = scene.place_new_component("resistor", QPointF(80.0, 0.0))
        assert r2 is not None
        assert len(scene.placed_items) == 2

        # Dragging R3 from (0, 120) onto half-overlapping (40, 0) must revert to (0, 120)
        r3 = scene.place_new_component("resistor", QPointF(0.0, 120.0))
        assert r3 is not None
        scene.begin_component_move_tracking()
        r3.setPos(QPointF(40.0, 0.0))
        assert scene.commit_component_move_tracking() is False
        assert (r3.pos().x(), r3.pos().y()) == (0.0, 120.0)
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_i2_overload_badge_populates_problems_panel_tooltip_and_inspector(qapp, tmp_path: Path) -> None:
    """I2: Component stress overload populates Problems panel, hover tooltip, and inspector banner."""
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "i2_settings"))
    try:
        scene = win.circuit_scene
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        r1 = scene.place_new_component("resistor", QPointF(0.0, 0.0))
        gnd = scene.place_new_component("ground", QPointF(0.0, 140.0))
        assert v1 is not None and r1 is not None and gnd is not None

        scene.edit_component_parameters(v1, {"dc_voltage": 12.0})
        # 12 V across 500 ohm (0.25 W rated) -> P = 144 / 500 = 288.0 mW (115.2% OVERLOAD)
        scene.edit_component_parameters(r1, {"resistance": 500.0})
        scene.add_wire(TerminalRef(v1.component.name, "pos"), TerminalRef(r1.component.name, "a"))
        scene.add_wire(TerminalRef(r1.component.name, "b"), TerminalRef(gnd.component.name, "gnd"))
        scene.add_wire(TerminalRef(v1.component.name, "neg"), TerminalRef(gnd.component.name, "gnd"))

        frame = SimulationSession(connectivity=scene.extract_connectivity_result()).start()
        win._on_simulation_frame(frame)
        qapp.processEvents()

        # (a) Problems panel lists R1 overload mentioning 288 mW and 250 mW
        tree = win.problems_dock.panel.tree
        assert tree.topLevelItemCount() >= 1
        r1_rows = [
            tree.topLevelItem(i)
            for i in range(tree.topLevelItemCount())
            if "R1" in tree.topLevelItem(i).text(2)
        ]
        assert r1_rows
        assert "288" in r1_rows[0].text(1) and "250" in r1_rows[0].text(1)

        # (b) Component tooltip contains 288 and overload
        tip = r1.toolTip().lower()
        assert "288" in tip and "overload" in tip

        # (c) Selecting R1 shows overload warning in Properties / Info panel
        scene.clearSelection()
        r1.setSelected(True)
        qapp.processEvents()
        info_txt = win.info_label.text().lower()
        assert "overload" in info_txt and "288" in info_txt

        # (d) Raising R1 to 1000 ohm (144 mW < 250 mW) clears Problems panel back to 0
        scene.edit_component_parameters(r1, {"resistance": 1_000.0})
        frame2 = SimulationSession(connectivity=scene.extract_connectivity_result()).start()
        win._on_simulation_frame(frame2)
        qapp.processEvents()
        assert tree.topLevelItemCount() == 0
        assert win.problems_dock.panel.count_label.text() == "(0)"
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()


def test_i3_light_theme_contrast_dock_title_and_synced_shortcuts_help(qapp, tmp_path, monkeypatch) -> None:
    """I3: Balanced matte Light Theme contrast, QDockWidget::title light styling, and dynamic shortcuts help."""
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.themes import LIGHT_PALETTE, ThemeMode, build_stylesheet, get_theme_palette

    monkeypatch.setenv("OHMBENCH_DATA_DIR", str(tmp_path))

    # (a) Distinct non-glaring Light Theme tokens and styled QDockWidget::title
    assert LIGHT_PALETTE.window_bg.upper() != "#FFFFFF"
    assert LIGHT_PALETTE.panel_bg.upper() != "#FFFFFF"
    assert LIGHT_PALETTE.canvas_bg.upper() != "#FFFFFF"
    assert len(
        {
            LIGHT_PALETTE.window_bg.upper(),
            LIGHT_PALETTE.panel_bg.upper(),
            LIGHT_PALETTE.canvas_bg.upper(),
        }
    ) == 3
    assert LIGHT_PALETTE.grid_minor.upper() != LIGHT_PALETTE.grid_major.upper()

    light_qss = build_stylesheet(LIGHT_PALETTE)
    assert "QDockWidget::title" in light_qss
    assert "#DCE2EA" in light_qss
    assert "#1E1E1E" not in light_qss

    # (b) Properties / Info panel shortcut text matches AppSettings.shortcuts dynamically
    win = MainWindow()
    try:
        win.apply_theme(ThemeMode.LIGHT)
        assert win.circuit_scene.theme == ThemeMode.LIGHT
        assert get_theme_palette(win.circuit_scene.theme) is LIGHT_PALETTE
        assert "#DCE2EA" in win.styleSheet()
        default_help = win.info_label.text()
        assert "Rotate 90° CW: Ctrl+R" in default_help
        assert "Rotate 90° CW: R\n" not in default_help
        assert "Resistor: R" in default_help

        win.settings.set_shortcut("rotate", "Alt+R")
        win.apply_shortcuts()
        qapp.processEvents()
        updated_help = win.info_label.text()
        assert "Rotate 90° CW: Alt+R" in updated_help
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()


def test_i4_ne555_dc_states_astable_oscillation_and_comparator() -> None:
    """I4: NE555 DC latch/discharge states, astable RC oscillation (3V..6V), and Comparator switching."""
    from circuitsim.components import (
        Capacitor,
        Comparator,
        DCVoltageSource,
        Ground,
        NE555,
        Resistor,
        create_default_component_model,
    )
    from circuitsim.engine.netlist import Circuit
    from circuitsim.engine.solver import MNASolver

    # Factory round-trip & G1 parameter validation
    assert create_default_component_model("ne555", "U1").component_type == "ne555"
    assert create_default_component_model("comparator", "U2").component_type == "comparator"

    # (a) 555 Timer DC latch & open-collector discharge test
    circ_dc = Circuit()
    vcc = DCVoltageSource("V_CC", "vcc", "0", voltage=9.0)
    vtrig = DCVoltageSource("V_TRIG", "trig", "0", voltage=2.0)
    vthr = DCVoltageSource("V_THR", "thr", "0", voltage=2.0)
    vrst = DCVoltageSource("V_RST", "rst", "0", voltage=9.0)
    r_pull = Resistor("R_PULL", "vcc", "dis", resistance=10_000.0)
    u555 = NE555("U1", "vcc", "0", "trig", "thr", "dis", "out", "rst")
    gnd = Ground("GND1", "0")
    for c in (vcc, vtrig, vthr, vrst, r_pull, u555, gnd):
        circ_dc.add(c)

    solver_dc = MNASolver(circ_dc)
    res1 = solver_dc.solve_dc()
    assert res1.node_voltages["out"] == pytest.approx(9.0, abs=0.2)
    assert res1.node_voltages["dis"] > 8.9  # DIS high-Z -> pulled to 9V

    # Hold inside [1/3 Vcc, 2/3 Vcc] = [3V, 6V]
    vtrig.dc_voltage = 4.5
    vthr.dc_voltage = 4.5
    res2 = solver_dc.solve_dc()
    assert res2.node_voltages["out"] == pytest.approx(9.0, abs=0.2)

    # Threshold > 6V -> OUT ~ 0V, DIS pulls 10k pull-up near 0V
    vthr.dc_voltage = 7.0
    res3 = solver_dc.solve_dc()
    assert res3.node_voltages["out"] < 0.1
    assert res3.node_voltages["dis"] < 0.05

    # Low RESET (< 0.7V) overrides TRIG (< 3V) and forces OUT low
    vtrig.dc_voltage = 2.0
    vrst.dc_voltage = 0.2
    res4 = solver_dc.solve_dc()
    assert res4.node_voltages["out"] < 0.1

    # (b) 555 Astable oscillator transient: VCC=9V, RA=1k, RB=2k, C=1uF -> oscillates across [3V, 6V]
    circ_ast = Circuit()
    circ_ast.add(DCVoltageSource("V1", "vcc", "0", voltage=9.0))
    circ_ast.add(Ground("G1", "0"))
    circ_ast.add(Resistor("R_A", "vcc", "dis", resistance=1_000.0))
    circ_ast.add(Resistor("R_B", "dis", "cap", resistance=2_000.0))
    circ_ast.add(Capacitor("C1", "cap", "0", capacitance=1e-6))
    circ_ast.add(NE555("U1", "vcc", "0", "cap", "cap", "dis", "out", "vcc"))

    tr = MNASolver(circ_ast).solve_transient(duration=6.5e-3, dt=1e-5)
    v_cap = tr.node_voltages["cap"]
    v_out = tr.node_voltages["out"]
    assert float(v_out.max()) > 8.8
    assert float(v_out.min()) < 0.2
    assert float(v_cap.max()) >= 5.95
    # Verify cap climbed to ~6V then discharged back down toward 3V
    idx_peak = int(v_cap.argmax())
    assert idx_peak < len(v_cap) - 10
    assert float(v_cap[idx_peak:].min()) <= 3.1

    # (c) Voltage Comparator switching between v_low=0V and v_high=5V
    circ_cmp = Circuit()
    vp = DCVoltageSource("VP", "np", "0", voltage=3.0)
    vn = DCVoltageSource("VN", "nn", "0", voltage=2.5)
    cmp_ic = Comparator("U_CMP", "np", "nn", "out", v_high=5.0, v_low=0.0)
    circ_cmp.add(vp)
    circ_cmp.add(vn)
    circ_cmp.add(cmp_ic)
    circ_cmp.add(Resistor("R_LOAD", "out", "0", resistance=10_000.0))
    circ_cmp.add(Ground("G_CMP", "0"))

    cmp_solver = MNASolver(circ_cmp)
    r_hi = cmp_solver.solve_dc()
    assert r_hi.node_voltages["out"] == pytest.approx(5.0, abs=0.05)

    vp.dc_voltage = 2.0
    r_lo = cmp_solver.solve_dc()
    assert r_lo.node_voltages["out"] == pytest.approx(0.0, abs=0.05)


def test_i5_transformer_spdt_relay_and_blowable_resettable_fuse() -> None:
    """I5: Coupled Transformer (2:1 AC step-down), SPDT Relay hysteresis, and blowable/resettable Fuse."""
    from circuitsim.components import (
        ACVoltageSource,
        DCVoltageSource,
        Fuse,
        Ground,
        Resistor,
        SPDTRelay,
        Transformer,
        create_default_component_model,
    )
    from circuitsim.engine.netlist import Circuit
    from circuitsim.engine.solver import MNASolver
    from circuitsim.engine.stress import StressLevel, evaluate_component_stress

    assert create_default_component_model("transformer", "T1").component_type == "transformer"
    assert create_default_component_model("spdt_relay", "K1").component_type == "spdt_relay"
    assert create_default_component_model("fuse", "F1").component_type == "fuse"

    # (a) 2:1 step-down Transformer: 20 V peak 100 Hz AC -> ~10.0 V peak on 1k secondary load
    circ_xf = Circuit()
    circ_xf.add(ACVoltageSource("V_AC", "p1", "0", amplitude=20.0, frequency=100.0))
    circ_xf.add(Ground("G_XF", "0"))
    circ_xf.add(Transformer("T1", "p1", "0", "s1", "0", turns_ratio=0.5))
    circ_xf.add(Resistor("R_SEC", "s1", "0", resistance=1_000.0))

    tr_xf = MNASolver(circ_xf).solve_transient(duration=15e-3, dt=1e-4)
    v_sec_peak = float(np.max(np.abs(tr_xf.node_voltages["s1"])))
    assert v_sec_peak == pytest.approx(10.0, rel=0.02)

    # (b) SPDT Relay: coil 0V -> NC energized (12V), NO ~ 0V; coil 6V -> NO energized (12V), NC ~ 0V
    circ_rel = Circuit()
    v_coil = DCVoltageSource("V_COIL", "coil_p", "0", voltage=0.0)
    circ_rel.add(v_coil)
    circ_rel.add(DCVoltageSource("V_BUS", "com", "0", voltage=12.0))
    circ_rel.add(Ground("G_REL", "0"))
    k1 = SPDTRelay("K1", "coil_p", "0", "com", "no", "nc", pull_in_voltage_v=5.0, drop_out_voltage_v=2.5)
    circ_rel.add(k1)
    circ_rel.add(Resistor("R_NO", "no", "0", resistance=100.0))
    circ_rel.add(Resistor("R_NC", "nc", "0", resistance=100.0))

    rel_solver = MNASolver(circ_rel)
    r_de = rel_solver.solve_dc()
    assert k1.energized is False
    assert r_de.node_voltages["nc"] == pytest.approx(12.0, abs=0.05)
    assert r_de.node_voltages["no"] < 0.05

    v_coil.dc_voltage = 6.0
    r_en = rel_solver.solve_dc()
    assert k1.energized is True
    assert r_en.node_voltages["no"] == pytest.approx(12.0, abs=0.05)
    assert r_en.node_voltages["nc"] < 0.05

    # Hysteresis: coil 3.5V (> 2.5V drop-out) holds energized state
    v_coil.dc_voltage = 3.5
    r_hold = rel_solver.solve_dc()
    assert k1.energized is True
    assert r_hold.node_voltages["no"] == pytest.approx(12.0, abs=0.05)

    # (c) 0.5 A Fuse on 12 V: intact with 100 ohm (0.12 A), blows with 10 ohm (1.2 A), reset restores
    circ_f = Circuit()
    circ_f.add(DCVoltageSource("V_F", "vin", "0", voltage=12.0))
    circ_f.add(Ground("G_F", "0"))
    f1 = Fuse("F1", "vin", "vload", rating_amps=0.5)
    r_load = Resistor("R_LOAD", "vload", "0", resistance=100.0)
    circ_f.add(f1)
    circ_f.add(r_load)

    f_solver = MNASolver(circ_f)
    res_ok = f_solver.solve_dc()
    assert f1.is_blown is False
    assert res_ok.branch_currents["R_LOAD"] == pytest.approx(0.12, abs=1e-3)

    r_load.resistance = 10.0  # Would draw 1.2 A > 0.5 A rating
    res_blown = f_solver.solve_dc()
    assert f1.is_blown is True
    assert abs(res_blown.branch_currents["R_LOAD"]) < 1e-6
    stress = evaluate_component_stress(circ_f, res_blown)["F1"]
    assert stress.level == StressLevel.OVERLOAD
    assert "blown" in stress.reason.lower()

    r_load.resistance = 100.0
    f1.reset_fuse()
    res_rst = f_solver.solve_dc()
    assert f1.is_blown is False
    assert res_rst.branch_currents["R_LOAD"] == pytest.approx(0.12, abs=1e-3)


def test_i6_sequential_d_jk_flipflops_4bit_counter_and_seven_seg_display(qapp) -> None:
    """I6: D/JK Flip-Flops, 4-bit Counter pulsed 5x driving 7-Segment Display ('5' = 0101)."""
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtCore import QRectF
    from circuitsim.components import (
        Counter4Bit,
        DCVoltageSource,
        DFlipFlop,
        Ground,
        JKFlipFlop,
        SevenSegDisplay,
        create_default_component_model,
    )
    from circuitsim.engine.netlist import Circuit
    from circuitsim.engine.solver import MNASolver

    for cid in ("d_flipflop", "jk_flipflop", "counter_4bit", "seven_seg_display"):
        assert create_default_component_model(cid, f"U_{cid}").component_type == cid

    # (a) D Flip-Flop & JK Flip-Flop
    circ_ff = Circuit()
    vd = DCVoltageSource("VD", "d", "0", voltage=5.0)
    vclk = DCVoltageSource("VCLK", "clk", "0", voltage=0.0)
    vrst = DCVoltageSource("VRST", "rst", "0", voltage=0.0)
    vj = DCVoltageSource("VJ", "j", "0", voltage=5.0)
    vk = DCVoltageSource("VK", "k", "0", voltage=5.0)
    dff = DFlipFlop("U_DFF", "d", "clk", "rst", "q_d", "qb_d")
    jkff = JKFlipFlop("U_JK", "j", "k", "clk", "rst", "q_jk", "qb_jk")
    for c in (vd, vclk, vrst, vj, vk, dff, jkff, Ground("G_FF", "0")):
        circ_ff.add(c)

    s_ff = MNASolver(circ_ff)
    r0 = s_ff.solve_dc()
    assert r0.node_voltages["q_d"] == pytest.approx(0.0, abs=0.05)

    # Rising clock edge: D latches 5V, JK toggles 0 -> 1
    vclk.dc_voltage = 5.0
    r1 = s_ff.solve_dc()
    assert r1.node_voltages["q_d"] == pytest.approx(5.0, abs=0.05)
    assert r1.node_voltages["qb_d"] == pytest.approx(0.0, abs=0.05)
    assert r1.node_voltages["q_jk"] == pytest.approx(5.0, abs=0.05)

    # Falling clock edge + D lowered to 0V -> D-FF holds 5V
    vclk.dc_voltage = 0.0
    vd.dc_voltage = 0.0
    r2 = s_ff.solve_dc()
    assert r2.node_voltages["q_d"] == pytest.approx(5.0, abs=0.05)

    # Second rising clock edge -> JK toggles 1 -> 0 while D latches 0V
    vclk.dc_voltage = 5.0
    r3 = s_ff.solve_dc()
    assert r3.node_voltages["q_d"] == pytest.approx(0.0, abs=0.05)
    assert r3.node_voltages["q_jk"] == pytest.approx(0.0, abs=0.05)

    # Async active-high RST clears Q=0V, Q_BAR=5V
    vclk.dc_voltage = 0.0
    vd.dc_voltage = 5.0
    s_ff.solve_dc()
    vclk.dc_voltage = 5.0
    assert s_ff.solve_dc().node_voltages["q_d"] == pytest.approx(5.0, abs=0.05)
    vrst.dc_voltage = 5.0
    r_rst = s_ff.solve_dc()
    assert r_rst.node_voltages["q_d"] == pytest.approx(0.0, abs=0.05)
    assert r_rst.node_voltages["qb_d"] == pytest.approx(5.0, abs=0.05)

    # (b) 4-Bit Binary Counter pulsed 5x wired to 7-Segment Display -> 0101 ('5')
    circ_cnt = Circuit()
    c_clk = DCVoltageSource("V_CK", "clk", "0", voltage=0.0)
    c_rst = DCVoltageSource("V_RS", "rst", "0", voltage=0.0)
    cnt = Counter4Bit("U_CNT", "clk", "rst", "q0", "q1", "q2", "q3")
    disp = SevenSegDisplay("DS1", "q0", "q1", "q2", "q3")
    for c in (c_clk, c_rst, cnt, disp, Ground("G_CNT", "0")):
        circ_cnt.add(c)

    s_cnt = MNASolver(circ_cnt)
    s_cnt.solve_dc()
    for _ in range(5):
        c_clk.dc_voltage = 5.0
        res_pulse = s_cnt.solve_dc()
        c_clk.dc_voltage = 0.0
        res_pulse = s_cnt.solve_dc()

    assert cnt.count == 5
    assert res_pulse.node_voltages["q0"] == pytest.approx(5.0, abs=0.05)
    assert res_pulse.node_voltages["q1"] == pytest.approx(0.0, abs=0.05)
    assert res_pulse.node_voltages["q2"] == pytest.approx(5.0, abs=0.05)
    assert res_pulse.node_voltages["q3"] == pytest.approx(0.0, abs=0.05)
    assert disp.decoded_hex_char == "5"
    assert set(disp.active_segments) == {"a", "f", "g", "c", "d"}

    # Render SevenSegDisplay in offscreen scene without error
    scene = CircuitScene(SymbolLibrary())
    item = scene.place_new_component("seven_seg_display", QPointF(0.0, 0.0))
    assert item is not None
    item.component.step_sequential({"DS1_D0": 5.0, "DS1_D1": 0.0, "DS1_D2": 5.0, "DS1_D3": 0.0})
    img = QImage(160, 180, QImage.Format.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    scene.render(painter, QRectF(0, 0, 160, 180), QRectF(-80, -90, 160, 180))
    painter.end()

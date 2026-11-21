"""Phase J Benchtop DSO-1062D, SPICE Exporter, Docs & Windows Packaging tests (J1-J6)."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.engine import MNASolver, SimulationSession
from circuitsim.engine.netlist import Circuit
from circuitsim.wiring.model import TerminalRef


def test_j1_phase_j_scaffolding(qapp) -> None:
    """J1: Verify offscreen Qt scaffolding and empty CircuitScene setup."""
    assert qapp is not None
    scene = CircuitScene(SymbolLibrary())
    try:
        assert scene.placed_items == []
        assert scene.wire_graph.wires == ()
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def _circle_intersects_rect(cx: float, cy: float, r: float, rect) -> bool:
    closest_x = max(rect.left(), min(cx, rect.right()))
    closest_y = max(rect.top(), min(cy, rect.bottom()))
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) < (r * r - 1e-6)


def test_j2_oscilloscope_component_and_dso_engine(qapp, tmp_path: Path) -> None:
    """J2: Verify Oscilloscope component, unwired-GND handling, DSO-1062D math/trigger/autoset, and .circ I/O."""
    from conftest import assert_module_has_no_qt_imports

    assert_module_has_no_qt_imports(
        "circuitsim.components.oscilloscope",
        "circuitsim.scope.dso_engine",
    )

    from circuitsim.components import ACVoltageSource, DCVoltageSource, Ground, Oscilloscope, Resistor
    from circuitsim.canvas.symbol_geometry import GRID_PITCH_PX, transform_terminal_position
    from circuitsim.io.share_codec import decode_circuit_share_url, encode_circuit_share_url
    from circuitsim.wiring.serializer import load_document_v2, save_document_v2
    from circuitsim.scope.dso_engine import (
        DSOState,
        compute_auto_set,
        compute_set_to_50_percent,
        process_dso_capture,
        recall_setup_json,
        reset_default_setup,
        save_setup_json,
    )

    # 1. Pure unwired-GND MNA probe check: must not trigger FloatingNodeError and reads exact 10 MOhm to GND
    c_unwired = Circuit()
    c_unwired.add(ACVoltageSource("V1", "n_sig", "0", amplitude=1.0, frequency=1000.0, dc_offset=5.0))
    c_unwired.add(Resistor("R1", "n_sig", "n_sig2", resistance=100.0))
    c_unwired.add(Resistor("R2", "n_sig2", "0", resistance=1000.0))
    osc_pure = Oscilloscope("OSC1", node_ch1="n_sig", node_ch2="n_sig2")
    c_unwired.add(osc_pure)
    c_unwired.add(Ground("G1", "0"))
    c_unwired.validate_topology()
    dc_res = MNASolver(c_unwired).solve_dc()
    assert dc_res.node_voltages[osc_pure.node_gnd] == pytest.approx(0.0, abs=1e-9)
    assert osc_pure.v_ch1 == pytest.approx(5.0, abs=1e-3)

    # 2. Place OSC1 on CircuitScene, wire CH1 (5V DC + 1V peak 1000Hz AC), CH2 (2V DC), GND
    scene = CircuitScene(SymbolLibrary())
    try:
        v_ac_item = scene.place_new_component("ac_voltage", QPointF(-200.0, -160.0))
        v_dc_item = scene.place_new_component("dc_voltage", QPointF(-200.0, 120.0))
        gnd_item = scene.place_new_component("ground", QPointF(-100.0, 240.0))
        osc_item = scene.place_new_component("oscilloscope", QPointF(160.0, 0.0))
        assert v_ac_item is not None and v_dc_item is not None
        assert gnd_item is not None and osc_item is not None

        v_ac_item.component.amplitude = 1.0
        v_ac_item.component.frequency = 1000.0
        v_ac_item.component.dc_offset = 5.0
        v_dc_item.component.dc_voltage = 2.0

        assert osc_item.component.component_type == "oscilloscope"
        assert osc_item.component.name == "OSC1"
        assert "Double-click to open DSO" in osc_item.get_live_tooltip()

        # Verify default_pin_label_rects clear of 5 px terminal dots across rotations 0, 90, 180, 270
        for rot in (0, 90, 180, 270):
            osc_item.set_rotation_deg(rot)
            dots = [
                transform_terminal_position(
                    t.grid_pos, (0.0, 0.0), rotation_deg=rot, flipped_h=False, grid_px=GRID_PITCH_PX
                )
                for t in osc_item.metadata.terminals
            ]
            label_rects = osc_item.default_pin_label_rects()
            assert len(label_rects) == 4
            for tname, tag, r in label_rects:
                for tx, ty in dots:
                    assert not _circle_intersects_rect(tx, ty, 5.0, r), (
                        f"OSC pin {tname}/{tag} overlaps terminal dot at rot={rot}"
                    )
        osc_item.set_rotation_deg(0)

        # Wire terminals cleanly
        vac_terms = v_ac_item.terminal_scene_positions_px()
        vdc_terms = v_dc_item.terminal_scene_positions_px()
        gnd_terms = gnd_item.terminal_scene_positions_px()
        osc_terms = osc_item.terminal_scene_positions_px()
        scene.add_routed_wire(vac_terms["pos"], osc_terms["CH1"], prefer_horizontal_first=False)
        scene.add_routed_wire(vdc_terms["pos"], osc_terms["CH2"], prefer_horizontal_first=False)
        scene.add_routed_wire(vac_terms["neg"], gnd_terms["gnd"], prefer_horizontal_first=True)
        scene.add_routed_wire(vdc_terms["neg"], gnd_terms["gnd"], prefer_horizontal_first=True)
        scene.add_routed_wire(osc_terms["GND"], gnd_terms["gnd"], prefer_horizontal_first=False)

        conn = scene.extract_connectivity_result()
        conn.circuit.validate_topology()
        solver = MNASolver(conn.circuit)
        tr = solver.solve_transient(duration=0.012, dt=1e-5)

        ch1_node = conn.component_nodes["OSC1"]["CH1"]
        ch2_node = conn.component_nodes["OSC1"]["CH2"]
        gnd_node = conn.component_nodes["OSC1"]["GND"]
        ch1_raw = tr.node_voltages[ch1_node] - tr.node_voltages[gnd_node]
        ch2_raw = tr.node_voltages[ch2_node] - tr.node_voltages[gnd_node]
        ext_raw = np.zeros_like(ch1_raw)

        # 2a. DC / AC / GND coupling
        state = DSOState()
        state.horizontal.sec_per_div = 1e-3  # 12 ms full view
        state.trigger.level_v = 5.0
        f_dc = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state)
        assert float(np.mean(f_dc.ch1_v)) == pytest.approx(5.0, abs=0.03)
        assert float(np.ptp(f_dc.ch1_v)) == pytest.approx(2.0, abs=0.05)

        state.ch1.coupling = "AC"
        f_ac = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state)
        assert float(np.mean(f_ac.ch1_v)) == pytest.approx(0.0, abs=0.03)
        assert float(np.ptp(f_ac.ch1_v)) == pytest.approx(2.0, abs=0.05)

        state.ch1.coupling = "GND"
        f_gnd = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state)
        assert np.allclose(f_gnd.ch1_v, 0.0, atol=1e-9)

        # 2b. MATH: CH1 - CH2 and FFT peak
        state.ch1.coupling = "DC"
        state.math.enabled = True
        state.math.operation = "CH1-CH2"
        state.math.fft_source = "CH1"
        f_math = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state)
        assert float(np.mean(f_math.math_v)) == pytest.approx(3.0, abs=0.03)
        assert f_math.fft_peak_hz is not None
        assert f_math.fft_peak_hz == pytest.approx(1000.0, abs=30.0)

        # 2c. SET TO 50%
        lvl_50 = compute_set_to_50_percent(ch1_raw)
        assert lvl_50 == pytest.approx(5.0, abs=0.05)
        state.trigger.level_v = lvl_50

        # 2d. AUTO SET centers trace inside 8 vertical divisions & shows ~2-4 cycles
        state_auto = compute_auto_set(tr.times, ch1_raw, ch2_raw, state)
        assert state_auto.trigger.level_v == pytest.approx(5.0, abs=0.05)
        assert state_auto.horizontal.sec_per_div in (200e-6, 500e-6)
        f_auto = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state_auto)
        div_vals = f_auto.ch1_v / state_auto.ch1.volts_per_div + state_auto.ch1.position_div
        assert abs(float(np.mean(div_vals))) <= 1.0
        assert float(np.max(np.abs(div_vals))) <= 4.0

        # 2e. CURSOR TIME & VOLTAGE
        state.cursors.cursor_mode = "TIME"
        state.cursors.t1 = 0.001
        state.cursors.t2 = 0.002
        state.cursors.v1 = 4.0
        state.cursors.v2 = 6.0
        f_cur = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state)
        assert f_cur.cursor_readout.delta_v == pytest.approx(2.0, abs=1e-9)
        assert f_cur.cursor_readout.delta_t == pytest.approx(0.001, abs=1e-9)
        assert f_cur.cursor_readout.inv_delta_t_hz == pytest.approx(1000.0, abs=1e-6)

        # 2f. NORMAL (wait), AUTO (free-run), SINGLE (one-shot & hold), FORCE TRIG
        state_trig = reset_default_setup()
        state_trig.trigger.mode = "NORMAL"
        state_trig.trigger.level_v = 10.0  # signal is 4..6 V -> no crossing
        f_wait = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state_trig)
        assert f_wait.triggered is False
        assert f_wait.trigger_status == "Wait"

        state_trig.trigger.mode = "AUTO"
        f_auto_free = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state_trig)
        assert f_auto_free.triggered is False
        assert f_auto_free.trigger_status == "Auto"

        state_trig.trigger.mode = "SINGLE"
        state_trig.trigger.level_v = 5.0
        state_trig.trigger.single_armed = True
        state_trig.running = True
        f_single1 = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state_trig)
        assert f_single1.triggered is True
        assert f_single1.captured_single is True
        assert state_trig.trigger.single_armed is False
        assert state_trig.running is False
        # Subsequent call holds cleanly without recapturing
        f_single2 = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state_trig, prev_frame=f_single1)
        assert f_single2.captured_single is False
        assert f_single2.trigger_status == "Stop"

        # FORCE TRIG forces capture even at 10V level
        state_trig.trigger.level_v = 10.0
        state_trig.trigger.forced = True
        f_forced = process_dso_capture(tr.times, ch1_raw, ch2_raw, ext_raw, state_trig, prev_frame=f_single2)
        assert f_forced.triggered is True
        assert f_forced.trigger_status == "Trig'd"

        # 2g. Save/Recall setup JSON, .circ v2 save/load, and share URL persistence
        setup_txt = save_setup_json(state_auto)
        osc_item.component.dso_state_json = setup_txt
        recalled = recall_setup_json(setup_txt)
        assert recalled.trigger.level_v == pytest.approx(state_auto.trigger.level_v, abs=1e-6)
        doc_v2 = scene.to_document_v2()
        circ_path = tmp_path / "osc_test.circ"
        save_document_v2(circ_path, doc_v2)
        loaded_doc = load_document_v2(circ_path)
        osc_specs = [p for p in loaded_doc.placed if p.component.component_type == "oscilloscope"]
        assert len(osc_specs) == 1
        assert recall_setup_json(osc_specs[0].component.dso_state_json).horizontal.sec_per_div == pytest.approx(
            state_auto.horizontal.sec_per_div
        )
        shared_url = encode_circuit_share_url(doc_v2)
        shared_doc = decode_circuit_share_url(shared_url)
        assert any(p.component.component_type == "oscilloscope" for p in shared_doc.placed)
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_j3_dso_1062d_front_panel_gui_and_exports(qapp, tmp_path: Path) -> None:
    """J3: Exercise Voltcraft DSO-1062D GUI menus, knobs, buttons, double-click dispatch & PNG/CSV exports."""
    from circuitsim.app.dso_panel import DSOOscilloscopeDialog
    from circuitsim.canvas.view import CircuitView

    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    opened_items: list[object] = []
    scene.oscilloscope_open_requested.connect(opened_items.append)
    dlg: DSOOscilloscopeDialog | None = None

    try:
        v_ac = scene.place_new_component("ac_voltage", QPointF(-240.0, -120.0))
        r1 = scene.place_new_component("resistor", QPointF(-100.0, 20.0))
        r2 = scene.place_new_component("resistor", QPointF(-100.0, 140.0))
        gnd = scene.place_new_component("ground", QPointF(-240.0, 240.0))
        osc = scene.place_new_component("oscilloscope", QPointF(180.0, 0.0))
        assert v_ac and r1 and r2 and gnd and osc

        v_ac.component.amplitude = 10.0
        v_ac.component.frequency = 1000.0
        v_ac.component.dc_offset = 0.0
        r1.component.resistance = 1000.0
        r2.component.resistance = 1000.0

        vac_t = v_ac.terminal_scene_positions_px()
        r1_t = r1.terminal_scene_positions_px()
        r2_t = r2.terminal_scene_positions_px()
        g_t = gnd.terminal_scene_positions_px()
        o_t = osc.terminal_scene_positions_px()

        # Half-amplitude resistive divider: V_ac -> R1 -> mid(R1.b/R2.a/CH2) -> R2 -> GND
        scene.add_routed_wire(vac_t["pos"], o_t["CH1"], prefer_horizontal_first=False)
        scene.add_routed_wire(vac_t["pos"], r1_t["a"], prefer_horizontal_first=False)
        scene.add_routed_wire(r1_t["b"], r2_t["a"], prefer_horizontal_first=True)
        scene.add_routed_wire(r1_t["b"], o_t["CH2"], prefer_horizontal_first=False)
        scene.add_routed_wire(r2_t["b"], g_t["gnd"], prefer_horizontal_first=True)
        scene.add_routed_wire(vac_t["neg"], g_t["gnd"], prefer_horizontal_first=True)
        scene.add_routed_wire(o_t["GND"], g_t["gnd"], prefer_horizontal_first=False)

        # Verify CircuitView double-click hook emits oscilloscope_open_requested
        assert view.open_quick_edit_for_item(osc) is True
        assert opened_items and opened_items[0] is osc
        for standalone in getattr(view, "_standalone_dso_dialogs", []):
            standalone.close()

        dlg = DSOOscilloscopeDialog(
            osc,
            circuit_provider=lambda: scene.extract_connectivity_result().circuit,
            nets_provider=lambda: scene.extract_connectivity_result().component_nodes.get("OSC1", {}),
        )
        dlg.show()
        frame = dlg.capture_from_scene()
        assert frame is not None
        assert frame.ch1_meas.v_pp == pytest.approx(20.0, abs=0.5)
        assert frame.ch2_meas.v_pp == pytest.approx(10.0, abs=0.3)
        assert len(osc.component.mini_preview_ch1) >= 8
        assert len(osc.component.mini_preview_ch2) >= 8

        # (a) Click all 11 menus & verify softkey sidebar updates + F0..F6 mutate state
        menu_checks = [
            (dlg.btn_ch1_menu, "CH1"),
            (dlg.btn_ch2_menu, "CH2"),
            (dlg.btn_math_menu, "MATH"),
            (dlg.btn_acquire, "ACQUIRE"),
            (dlg.btn_measure, "MEASURE"),
            (dlg.btn_cursor, "CURSOR"),
            (dlg.btn_display, "DISPLAY"),
            (dlg.btn_horiz_menu, "HORIZ"),
            (dlg.btn_trig_menu, "TRIG"),
            (dlg.btn_save_recall, "SAVE/RECALL"),
            (dlg.btn_utility, "UTILITY"),
        ]
        for btn, title in menu_checks:
            btn.click()
            assert dlg.screen_widget.active_menu_title == title
            assert len(dlg.screen_widget.menu_rows) >= 4

        dlg.btn_ch1_menu.click()
        dlg.softkey_buttons[1].click()  # DC -> AC
        assert dlg.state.ch1.coupling == "AC"
        dlg.softkey_buttons[1].click()
        dlg.softkey_buttons[1].click()  # GND -> DC
        assert dlg.state.ch1.coupling == "DC"

        dlg.btn_math_menu.click()
        dlg.softkey_buttons[0].click()  # Enable Math
        assert dlg.state.math.enabled is True
        dlg.softkey_buttons[1].click()  # Cycle operator
        assert dlg.state.math.operation != "CH1-CH2"

        dlg.btn_cursor.click()
        dlg.softkey_buttons[0].click()  # OFF -> VOLTAGE
        assert dlg.state.cursors.cursor_mode == "VOLTAGE"

        dlg.btn_display.click()
        dlg.softkey_buttons[0].click()  # VECTORS -> DOTS
        assert dlg.state.display.render_style == "DOTS"

        dlg.btn_acquire.click()
        dlg.softkey_buttons[2].click()  # 4 -> 16 averages
        assert dlg.state.acquire.avg_count == 16

        # (b) Front-panel run/trigger/default buttons
        dlg.btn_auto_set.click()
        assert dlg.state.running is True
        dlg.btn_set_50.click()
        assert abs(dlg.state.trigger.level_v) < 0.25
        dlg.btn_force_trig.click()
        dlg.btn_single_seq.click()
        assert dlg.state.trigger.mode == "SINGLE"
        dlg.btn_run_stop.click()
        assert dlg.state.running is False
        dlg.btn_default_setup.click()
        assert dlg.state.math.enabled is False
        assert dlg.state.display.render_style == "VECTORS"

        # (c) PNG screenshot & CSV waveform export
        png_path = tmp_path / "dso.png"
        csv_path = tmp_path / "dso.csv"
        dlg.export_screen_png(png_path)
        dlg.export_waveform_csv(csv_path)

        assert png_path.exists() and png_path.stat().st_size > 200
        assert png_path.read_bytes().startswith(b"\x89PNG")
        assert csv_path.exists() and csv_path.stat().st_size > 100
        csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert csv_lines[0] == "time_s,ch1_v,ch2_v,math_v"
        assert len(csv_lines) > 20
    finally:
        if dlg is not None:
            dlg.close()
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()


def test_j4_properties_dialog_vip_inductor_and_lightbulb_helpers(qapp) -> None:
    """J4: Verify live V/I/P banner, Inductor X_L/tau & reverse calc, and Lightbulb R_hot in QuickValueDialog."""
    from circuitsim.components import Inductor, Lightbulb
    from circuitsim.dialogs.quick_edit_dialog import (
        QuickValueDialog,
        compute_inductance_from_reactance,
        compute_inductor_reactance,
        compute_inductor_tau,
    )
    from circuitsim.dialogs.value_parser import parse_engineering_value

    assert compute_inductor_reactance(0.01, 1000.0) == pytest.approx(62.831853, rel=1e-4)
    assert compute_inductor_tau(0.01, 100.0) == pytest.approx(100e-6, rel=1e-6)
    assert compute_inductance_from_reactance(314.159265, 1000.0) == pytest.approx(0.05, rel=1e-4)

    ind = Inductor("L1", "n1", "0", inductance=0.01, dcr=0.5)
    ind.voltage = 5.0
    ind.current = 0.25
    ind.power = 1.25
    dlg_ind = QuickValueDialog(ind)
    try:
        vip_txt = dlg_ind.live_vip_label.text()
        assert "5 V" in vip_txt
        assert "250 mA" in vip_txt
        assert "1.25 W" in vip_txt

        assert dlg_ind.inductor_freq_input is not None
        assert dlg_ind.inductor_r_total_input is not None
        assert dlg_ind.inductor_xl_label is not None
        assert dlg_ind.inductor_tau_label is not None
        assert dlg_ind.inductor_target_xl_input is not None
        assert dlg_ind.inductor_reverse_calc_btn is not None

        dlg_ind.inductor_freq_input.setText("1000")
        dlg_ind.inductor_r_total_input.setText("100")
        assert "62.83" in dlg_ind.inductor_xl_label.text()
        assert "100" in dlg_ind.inductor_tau_label.text()

        dlg_ind.inductor_target_xl_input.setText("314.159265")
        dlg_ind.inductor_reverse_calc_btn.click()
        l_parsed = parse_engineering_value(dlg_ind.inputs["inductance"].text())
        assert l_parsed == pytest.approx(0.05, rel=1e-4)
    finally:
        dlg_ind.close()

    bulb = Lightbulb("B1", "n1", "0", rated_power=60.0, rated_voltage=120.0)
    bulb.voltage = 120.0
    bulb.current = 0.5
    bulb.power = 60.0
    dlg_bulb = QuickValueDialog(bulb)
    try:
        assert "rated_power" in dlg_bulb.inputs
        assert "rated_voltage" in dlg_bulb.inputs
        assert dlg_bulb.lightbulb_r_hot_label is not None
        assert "240" in dlg_bulb.lightbulb_r_hot_label.text()
        assert "60 W" in dlg_bulb.live_vip_label.text()
    finally:
        dlg_bulb.close()
        qapp.processEvents()


def test_j5_spice_netlist_exporter(qapp) -> None:
    """J5: Verify Qt-free SPICE netlist exporter with MEG/m/u/k suffixes, .MODEL, and virtual instrument comments."""
    from conftest import assert_module_has_no_qt_imports

    assert_module_has_no_qt_imports("circuitsim.io.spice_export")

    from circuitsim.io.spice_export import export_to_spice_netlist, format_spice_value

    assert format_spice_value(1_000_000.0) == "1MEG"
    assert format_spice_value(0.001) == "1m"
    assert format_spice_value(1e-6) == "1u"
    assert format_spice_value(4700.0) == "4.7k"
    assert format_spice_value(0.015) == "15m"

    scene = CircuitScene(SymbolLibrary())
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-300.0, -100.0))
        r1 = scene.place_new_component("resistor", QPointF(-140.0, -100.0))
        r2 = scene.place_new_component("resistor", QPointF(-140.0, 20.0))
        d1 = scene.place_new_component("diode", QPointF(20.0, -100.0))
        q1 = scene.place_new_component("npn_bjt", QPointF(160.0, -20.0))
        osc = scene.place_new_component("oscilloscope", QPointF(360.0, 0.0))
        gnd = scene.place_new_component("ground", QPointF(-220.0, 180.0))
        assert v1 and r1 and r2 and d1 and q1 and osc and gnd

        v1.component.dc_voltage = 12.0
        r1.component.resistance = 1_000_000.0
        r2.component.resistance = 0.015

        vt = v1.terminal_scene_positions_px()
        r1t = r1.terminal_scene_positions_px()
        r2t = r2.terminal_scene_positions_px()
        d1t = d1.terminal_scene_positions_px()
        q1t = q1.terminal_scene_positions_px()
        ot = osc.terminal_scene_positions_px()
        gt = gnd.terminal_scene_positions_px()

        scene.add_routed_wire(vt["pos"], r1t["a"], prefer_horizontal_first=False)
        scene.add_routed_wire(r1t["b"], d1t["anode"], prefer_horizontal_first=True)
        scene.add_routed_wire(d1t["cathode"], q1t["B"], prefer_horizontal_first=False)
        scene.add_routed_wire(vt["pos"], q1t["C"], prefer_horizontal_first=False)
        scene.add_routed_wire(q1t["E"], r2t["a"], prefer_horizontal_first=False)
        scene.add_routed_wire(r2t["b"], gt["gnd"], prefer_horizontal_first=False)
        scene.add_routed_wire(vt["neg"], gt["gnd"], prefer_horizontal_first=True)
        scene.add_routed_wire(q1t["B"], ot["CH1"], prefer_horizontal_first=False)
        scene.add_routed_wire(ot["GND"], gt["gnd"], prefer_horizontal_first=False)

        # Label one wire VCC explicitly
        vcc_wire = next(iter(scene.wire_graph.wires))
        scene._wire_graph = scene.wire_graph.set_wire_net_label(vcc_wire.id, "VCC")

        cir = export_to_spice_netlist(scene, title="Mixed J5 Test")
        assert "* Mixed J5 Test" in cir
        assert "1MEG" in cir
        assert "15m" in cir
        assert " 0 " in cir or " 0\n" in cir
        assert ".MODEL" in cir
        assert "* Oscilloscope OSC1" in cir
        assert cir.rstrip().endswith(".end")
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_j6_packaging_assets_resource_helper_and_docs(monkeypatch, tmp_path: Path) -> None:
    """J6: Verify packaging specs, frozen resource resolution, docs, and .gitignore rules."""
    import sys

    from circuitsim.canvas.symbol_geometry import get_resource_path

    repo_root = Path(__file__).resolve().parent.parent

    spec_path = repo_root / "packaging" / "ohmbench.spec"
    iss_path = repo_root / "packaging" / "installer.iss"
    assert spec_path.is_file()
    assert iss_path.is_file()

    spec_txt = spec_path.read_text(encoding="utf-8")
    assert "exclude_binaries=True" in spec_txt
    assert "COLLECT(" in spec_txt
    assert "console=False" in spec_txt
    assert 'name="Ohmbench"' in spec_txt or "name='Ohmbench'" in spec_txt
    assert (repo_root / "src" / "circuitsim" / "main.py").is_file()
    assert (repo_root / "src" / "circuitsim" / "resources").is_dir()
    assert (repo_root / "src" / "circuitsim" / "templates").is_dir()
    assert "resources" in spec_txt and "templates" in spec_txt

    iss_txt = iss_path.read_text(encoding="utf-8")
    assert r"..\dist\Ohmbench\*" in iss_txt
    assert "{group}" in iss_txt
    assert "desktopicon" in iss_txt
    assert r"..\LICENSE" in iss_txt
    assert (repo_root / "LICENSE").is_file()

    # Non-frozen & monkeypatched PyInstaller _MEIPASS resolution
    iec_dir = get_resource_path("resources", "symbols", "iec")
    assert iec_dir.is_dir() and (iec_dir / "oscilloscope.json").is_file()

    fake_iec = tmp_path / "circuitsim" / "resources" / "symbols" / "iec"
    fake_iec.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert get_resource_path("resources", "symbols", "iec") == fake_iec

    readme_txt = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "Building and Packaging (Windows)" in readme_txt
    assert "--onedir" in readme_txt
    assert "PySide6" in readme_txt
    assert "LGPL" in readme_txt
    assert "BSD 3-Clause" in readme_txt or "BSD-3-Clause" in readme_txt

    gitignore_txt = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "dist/" in gitignore_txt
    assert "build/" in gitignore_txt
    assert "Output/" in gitignore_txt
    assert "!packaging/ohmbench.spec" in gitignore_txt

    learn_txt = (repo_root / "LEARNING_NOTES.md").read_text(encoding="utf-8")
    assert "DSO-1062D" in learn_txt
    assert "WireGraph" in learn_txt
    assert "Three Hands-On Student Exercises" in learn_txt

    sugg_txt = (repo_root / "SUGGESTIONS.md").read_text(encoding="utf-8")
    assert "Post-v1.0 Prioritized Roadmap" in sugg_txt
    assert sugg_txt.count("*(Effort:") >= 5


def test_l1_50hz_rc_high_rate_smooth_capture_no_aliasing(qapp) -> None:
    """L1: Stream 50Hz 60Vpp RC filter via SampleBlock sink without UI-frame aliasing."""
    from circuitsim.app.dso_panel import DSOOscilloscopeDialog
    from circuitsim.components import ACVoltageSource, Capacitor, Ground, Oscilloscope, Resistor

    circuit = Circuit()
    circuit.add(ACVoltageSource("V1", "n_in", "0", amplitude=30.0, frequency=50.0, dc_offset=0.0))
    circuit.add(Resistor("R1", "n_in", "n_cap", resistance=1000.0))
    circuit.add(Capacitor("C1", "n_cap", "0", capacitance=10e-6))
    circuit.add(Ground("G1", "0"))
    osc = Oscilloscope("OSC1", node_ch1="n_cap", node_gnd="0")
    circuit.add(osc)
    circuit.validate_topology()

    dlg = DSOOscilloscopeDialog(osc)
    try:
        dlg.state.horizontal.sec_per_div = 5e-3
        session = SimulationSession(circuit=circuit)
        session.set_timebase(5e-3, points_per_div=100)
        session.attach_scope_sink(dlg)
        session.start()
        for _ in range(8):
            session.step_chunk(steps=350)

        frame = dlg.latest_frame
        assert frame is not None
        assert len(frame.ch1_v) >= 150
        assert frame.ch1_measurements.v_pp == pytest.approx(18.18, rel=0.08)
        assert frame.ch1_measurements.freq_hz == pytest.approx(50.0, abs=0.5)
        assert frame.hw_freq_counter_hz == pytest.approx(50.0, abs=0.5)
    finally:
        dlg.close()
        qapp.processEvents()


def test_l2_exact_grid_division_width_and_roll_mode_no_clumping(qapp) -> None:
    """L2: Enforce exact sec/div pixel scaling (no stretch of partial captures) and right-anchored Roll Mode."""
    from circuitsim.scope.dso_engine import DSOState, process_dso_capture

    # 1. Full 50 Hz sine at 5 ms/div -> 20 ms period == exact 4.000 grid divisions
    t_full = np.linspace(0.0, 0.12, 2401, dtype=np.float64)
    ch1_full = 5.0 * np.sin(2.0 * np.pi * 50.0 * t_full)
    zeros_full = np.zeros_like(ch1_full)

    state = DSOState()
    state.horizontal.sec_per_div = 5e-3
    f_full = process_dso_capture(t_full, ch1_full, zeros_full, None, state)
    assert f_full.roll_mode is False
    rise_idx = np.flatnonzero((f_full.ch1_v[:-1] < 0.0) & (f_full.ch1_v[1:] >= 0.0))
    assert len(rise_idx) >= 2
    cross_x = []
    for idx in rise_idx:
        dv = f_full.ch1_v[idx + 1] - f_full.ch1_v[idx]
        frac = -f_full.ch1_v[idx] / dv if abs(dv) > 1e-12 else 0.0
        cross_x.append((f_full.x_norm[idx] + frac * (f_full.x_norm[idx + 1] - f_full.x_norm[idx])) * 12.0)
    div_spacings = np.diff(cross_x)
    assert len(div_spacings) >= 1
    assert np.allclose(div_spacings, 4.0, atol=0.05)

    # 2. Partial 15 ms capture at 5 ms/div -> must occupy 3.0 divisions, NOT stretch to 12.0
    t_part = np.linspace(0.0, 0.015, 301, dtype=np.float64)
    ch1_part = 5.0 * np.sin(2.0 * np.pi * 50.0 * t_part)
    f_part = process_dso_capture(t_part, ch1_part, np.zeros_like(ch1_part), None, state)
    width_divs = float(f_part.x_norm[-1] - f_part.x_norm[0]) * 12.0
    assert width_divs == pytest.approx(3.0, abs=0.05)

    # 3. Roll Mode (>= 100 ms/div): 1.0 s of data at 1 s/div -> [11/12, 1.0] right-anchored
    state.horizontal.sec_per_div = 1.0
    t_roll = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    ch1_roll = 5.0 * np.sin(2.0 * np.pi * 1.0 * t_roll)
    f_roll = process_dso_capture(t_roll, ch1_roll, np.zeros_like(ch1_roll), None, state)
    assert f_roll.roll_mode is True
    assert float(f_roll.x_norm[-1]) == pytest.approx(1.0, abs=1e-6)
    assert float(f_roll.x_norm[0]) == pytest.approx(11.0 / 12.0, abs=1e-3)


def test_l3_run_stop_spam_preserves_valid_waveform_and_auto_set(qapp) -> None:
    """L3: Rapid RUN/STOP toggles preserve waveform integrity; AUTO SET selects 5 ms/div & 5 V/div."""
    from circuitsim.app.dso_panel import DSOOscilloscopeDialog
    from circuitsim.components import ACVoltageSource, Capacitor, Ground, Oscilloscope, Resistor

    circuit = Circuit()
    circuit.add(ACVoltageSource("V1", "n_in", "0", amplitude=30.0, frequency=50.0, dc_offset=0.0))
    circuit.add(Resistor("R1", "n_in", "n_cap", resistance=1000.0))
    circuit.add(Capacitor("C1", "n_cap", "0", capacitance=10e-6))
    circuit.add(Ground("G1", "0"))
    osc = Oscilloscope("OSC1", node_ch1="n_cap", node_gnd="0")
    circuit.add(osc)
    circuit.validate_topology()

    dlg = DSOOscilloscopeDialog(osc)
    try:
        dlg.state.horizontal.sec_per_div = 5e-3
        session = SimulationSession(circuit=circuit)
        session.set_timebase(5e-3, points_per_div=100)
        session.attach_scope_sink(dlg)
        session.start()
        for _ in range(8):
            session.step_chunk(steps=350)

        assert dlg.latest_frame is not None
        assert len(dlg.latest_frame.ch1_v) >= 50
        assert dlg.latest_frame.ch1_measurements.v_pp > 15.0

        # Spam RUN/STOP 11 times -> ends stopped & preserves captured waveform
        for _ in range(11):
            dlg.btn_run_stop.click()
            qapp.processEvents()
        assert dlg.state.running is False
        assert dlg.latest_frame is not None
        assert len(dlg.latest_frame.ch1_v) >= 50
        assert dlg.latest_frame.ch1_measurements.v_pp > 15.0

        # Resume and trigger AUTO SET
        dlg.btn_auto_set.click()
        qapp.processEvents()
        assert dlg.state.running is True
        assert dlg.state.horizontal.sec_per_div == pytest.approx(5e-3)
        assert dlg.state.ch1.volts_per_div == pytest.approx(5.0)
        assert abs(dlg.state.ch1.position_div) < 0.5
    finally:
        dlg.close()
        qapp.processEvents()


def test_l4_dso_layout_unclipped_and_timebase_text_complete(qapp) -> None:
    """L4: Bottom BNC strip stays inside dialog bounds and timebase text includes '/div'."""
    from PySide6.QtCore import QPoint, QRect

    from circuitsim.app.dso_panel import DSOOscilloscopeDialog
    from circuitsim.components import Oscilloscope

    dlg = DSOOscilloscopeDialog(Oscilloscope("OSC1"))
    try:
        dlg.show()
        qapp.processEvents()

        dlg_rect = dlg.rect()
        for widget in (dlg.btn_run_stop, dlg.btn_auto_set, dlg.bnc_bar, *dlg.bnc_labels.values()):
            mapped = QRect(widget.mapTo(dlg, QPoint(0, 0)), widget.size())
            assert dlg_rect.contains(mapped), f"Widget {widget.objectName() or widget.text()} clipped: {mapped} outside {dlg_rect}"

        dlg.screen_widget.grab()
        assert dlg.screen_widget.top_timebase_text.endswith("/div")
        assert "M 5" in dlg.screen_widget.top_timebase_text

        dlg.state.horizontal.sec_per_div = 1.0
        dlg._sync_ui_from_state()
        dlg.screen_widget.grab()
        assert dlg.screen_widget.top_timebase_text == "M 1s/div"
    finally:
        dlg.close()
        qapp.processEvents()

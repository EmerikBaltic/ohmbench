"""Phase M regression suite: DSO polish, Function Generator, AC RMS meters, Logic Pack, Quick-Place & Custom Parts Library."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_m0_phase_m_scaffolding(qapp: QApplication) -> None:
    assert qapp is not None


def test_m1_dso_autoset_startup_ramp_and_softkey_formatting(qapp: QApplication) -> None:
    from circuitsim.app.dso_panel import DSOOscilloscopeDialog, format_sdiv, format_vdiv
    from circuitsim.components.capacitor import Capacitor
    from circuitsim.components.oscilloscope import Oscilloscope
    from circuitsim.components.resistor import Resistor
    from circuitsim.components.sources import ACVoltageSource
    from circuitsim.engine.netlist import Circuit
    from circuitsim.scope.dso_engine import (
        DSO_SEC_PER_DIV,
        DSO_VOLTS_PER_DIV,
        DSOState,
        compute_auto_set,
    )

    assert format_vdiv(0.1) == "100 mV"
    assert format_vdiv(0.5) == "500 mV"
    assert format_sdiv(0.002) == "2 ms"

    # Compute auto set without circuit provider on a 30 us linear ramp -> must not shrink to us/div
    t_ramp = np.linspace(0.0, 30e-6, 32)
    c1_ramp = np.linspace(0.0, 0.075, 32)
    c2_ramp = np.zeros_like(t_ramp)
    ramp_state = compute_auto_set(t_ramp, c1_ramp, c2_ramp, DSOState())
    assert ramp_state.horizontal.sec_per_div >= 5e-3

    # Build 10 Vpk, 50 Hz AC source -> 1 kOhm -> 1 uF || Oscilloscope
    def _build_rc_circuit() -> tuple[Circuit, Oscilloscope]:
        from circuitsim.components.ground import Ground

        c = Circuit()
        src = ACVoltageSource("V1", "vin", "0", amplitude=10.0, frequency=50.0)
        r1 = Resistor("R1", "vin", "vout", resistance=1000.0)
        cap = Capacitor("C1", "vout", "0", capacitance=1e-6)
        osc = Oscilloscope("OSC1", node_ch1="vout", node_gnd="0")
        gnd = Ground("GND", "0")
        c.add(src)
        c.add(r1)
        c.add(cap)
        c.add(osc)
        c.add(gnd)
        return c, osc

    _, osc_ref = _build_rc_circuit()
    dlg = DSOOscilloscopeDialog(
        osc_ref,
        circuit_provider=lambda: _build_rc_circuit()[0],
    )
    try:
        # Seed ring buffer with only 30 us of startup ramp (0..75 mV)
        dlg._ring_buffer.set_arrays(t_ramp, c1_ramp, c2_ramp, c2_ramp)
        dlg._on_auto_set()

        assert dlg.state.ch1.volts_per_div in (2.0, 5.0)
        assert dlg.state.horizontal.sec_per_div in (5e-3, 10e-3)
        assert dlg.latest_frame is not None
        assert dlg.latest_frame.ch1_meas.v_pp > 10.0

        # Check all V/div and s/div softkey labels have zero scientific notation
        for vdiv in DSO_VOLTS_PER_DIV:
            dlg.state.ch1.volts_per_div = vdiv
            dlg.active_menu = "CH1"
            for label, val in dlg._menu_options():
                assert "e+" not in val and "e-" not in val, f"Bad V/div softkey {label}={val}"

        for sdiv in DSO_SEC_PER_DIV:
            dlg.state.horizontal.sec_per_div = sdiv
            dlg.active_menu = "HORIZ"
            for label, val in dlg._menu_options():
                assert "e+" not in val and "e-" not in val, f"Bad s/div softkey {label}={val}"
    finally:
        dlg.close()


def test_m2_function_generator_physics_impedances_gui_and_serialization(
    qapp: QApplication, tmp_path
) -> None:
    from PySide6.QtCore import QPointF, Qt
    from circuitsim.app.funcgen_panel import FunctionGeneratorDialog
    from circuitsim.app.palette_dock import ComponentPaletteWidget
    from circuitsim.canvas.scene import CircuitScene
    from circuitsim.canvas.symbol_renderer import SymbolLibrary
    from circuitsim.components.function_generator import FunctionGenerator
    from circuitsim.components.ground import Ground
    from circuitsim.components.resistor import Resistor
    from circuitsim.components.sources import WaveformType
    from circuitsim.engine.netlist import Circuit
    from circuitsim.engine.solver import MNASolver
    from circuitsim.io.circ_serializer import (
        CircuitDocument,
        PlacedComponentSpec,
        load_document,
        save_document,
    )

    def _solve_fg(
        *,
        waveform: str,
        freq: float = 100.0,
        amp: float = 5.0,
        offset: float = 0.0,
        duty: float = 0.5,
        zout: float = 0.0,
        enabled: bool = True,
        r_load: float = 1000.0,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        c = Circuit()
        fg = FunctionGenerator(
            "FG1",
            "vout",
            "0",
            waveform=waveform,
            frequency=freq,
            amplitude=amp,
            dc_offset=offset,
            duty_cycle=duty,
            output_impedance=zout,
            output_enabled=enabled,
        )
        r = Resistor("R1", "vout", "0", resistance=r_load)
        g = Ground("GND", "0")
        c.add(fg)
        c.add(r)
        c.add(g)
        dc = MNASolver(c).solve_dc()
        tr = MNASolver(c).solve_transient(duration=2.0 / freq, dt=1.0 / (200.0 * freq))
        return float(dc.node_voltages["vout"]), tr.times, tr.node_voltages["vout"]

    # 1. All 4 waveforms + duty cycle + DC offset
    dc_sine, _, v_sine = _solve_fg(waveform="sine", amp=5.0, offset=1.2)
    assert dc_sine == pytest.approx(1.2, abs=1e-4)
    assert np.max(v_sine) == pytest.approx(6.2, abs=0.05)
    assert np.min(v_sine) == pytest.approx(-3.8, abs=0.05)

    dc_sq, _, v_sq = _solve_fg(waveform="square", amp=5.0, offset=0.0, duty=0.25)
    assert dc_sq == pytest.approx(0.0, abs=1e-4)
    high_frac = float(np.mean(v_sq > 0.0))
    assert high_frac == pytest.approx(0.25, abs=0.04)

    _, _, v_tri = _solve_fg(waveform="triangle", amp=4.0, duty=0.5)
    assert np.max(v_tri) == pytest.approx(4.0, abs=0.1)
    assert np.min(v_tri) == pytest.approx(-4.0, abs=0.1)

    _, _, v_saw = _solve_fg(waveform="sawtooth", amp=4.0)
    assert np.max(v_saw) == pytest.approx(4.0, abs=0.15)
    assert np.min(v_saw) == pytest.approx(-4.0, abs=0.15)

    # 2. Output impedance: Hi-Z (0 Ohm) vs 50 Ohm into 50 Ohm load -> 50% divider
    _, _, v_hiz = _solve_fg(waveform="sine", amp=5.0, zout=0.0, r_load=50.0)
    _, _, v_50 = _solve_fg(waveform="sine", amp=5.0, zout=50.0, r_load=50.0)
    assert np.max(v_hiz) == pytest.approx(5.0, abs=0.05)
    assert np.max(v_50) == pytest.approx(2.5, abs=0.05)

    # 3. Output disabled -> 0.0 V
    dc_off, _, v_off = _solve_fg(waveform="square", amp=5.0, offset=2.0, enabled=False)
    assert dc_off == pytest.approx(0.0, abs=1e-6)
    assert np.max(np.abs(v_off)) == pytest.approx(0.0, abs=1e-6)

    # 4. Palette placement under Sources & Instruments + scene pin labels
    lib = SymbolLibrary()
    palette = ComponentPaletteWidget(lib)
    try:
        cats_with_fg: list[str] = []
        for i in range(palette.tree.topLevelItemCount()):
            cat_item = palette.tree.topLevelItem(i)
            for j in range(cat_item.childCount()):
                if cat_item.child(j).data(0, Qt.ItemDataRole.UserRole) == "function_generator":
                    cats_with_fg.append(cat_item.text(0))
        assert "Sources" in cats_with_fg
        assert "Instruments" in cats_with_fg
    finally:
        palette.close()

    scene = CircuitScene(lib)
    item = scene.place_new_component("function_generator", QPointF(0.0, 0.0))
    assert item.component.name == "FG1"
    assert len(item.metadata.terminals) == 2
    pin_rects = item.default_pin_label_rects()
    assert {tag for _t, tag, _r in pin_rects} == {"OUT", "GND"}

    # 5. Modeless dialog updates component & serialization round-trip
    fg = item.component
    assert isinstance(fg, FunctionGenerator)
    dlg = FunctionGeneratorDialog(item)
    try:
        dlg.waveform_buttons[WaveformType.TRIANGLE].click()
        dlg._set_freq_unit(1e3)
        dlg.freq_spin.setValue(2.5)
        dlg.amp_spin.setValue(3.3)
        dlg.offset_spin.setValue(0.5)
        dlg.duty_spin.setValue(30.0)
        dlg.zout_btn.click()  # toggle to 50 Ohm
        dlg.output_btn.click()  # toggle off

        assert fg.waveform == WaveformType.TRIANGLE
        assert fg.frequency == pytest.approx(2500.0)
        assert fg.amplitude == pytest.approx(3.3)
        assert fg.dc_offset == pytest.approx(0.5)
        assert fg.duty_cycle == pytest.approx(0.30)
        assert fg.output_impedance == pytest.approx(50.0)
        assert fg.output_enabled is False
    finally:
        dlg.close()

    doc = CircuitDocument(
        placed=[PlacedComponentSpec(component=fg, center_px=(0.0, 0.0))]
    )
    circ_path = tmp_path / "fg_roundtrip.circ"
    save_document(circ_path, doc)
    loaded = load_document(circ_path)
    fg_loaded = loaded.placed[0].component
    assert isinstance(fg_loaded, FunctionGenerator)
    assert fg_loaded.waveform == WaveformType.TRIANGLE
    assert fg_loaded.frequency == pytest.approx(2500.0)
    assert fg_loaded.amplitude == pytest.approx(3.3)
    assert fg_loaded.dc_offset == pytest.approx(0.5)
    assert fg_loaded.duty_cycle == pytest.approx(0.30)
    assert fg_loaded.output_impedance == pytest.approx(50.0)
    assert fg_loaded.output_enabled is False


def test_m3_ac_rms_voltmeter_and_ammeter_telemetry_and_serialization(
    qapp: QApplication, tmp_path
) -> None:
    import math
    from circuitsim.components.ground import Ground
    from circuitsim.components.meters import Ammeter, MeterMode, Voltmeter
    from circuitsim.components.resistor import Resistor
    from circuitsim.components.sources import ACVoltageSource
    from circuitsim.dialogs.quick_edit_dialog import QuickValueDialog
    from circuitsim.engine.netlist import Circuit
    from circuitsim.engine.simulation_session import SimulationSession
    from circuitsim.io.circ_serializer import (
        CircuitDocument,
        PlacedComponentSpec,
        load_document,
        save_document,
    )

    c = Circuit()
    src = ACVoltageSource("V1", "vin", "0", amplitude=10.0, frequency=100.0)
    am = Ammeter("A1", "vin", "vload", is_ideal=True, measurement_mode="AC_RMS", window_duration=0.05)
    vm = Voltmeter("VM1", "vload", "0", is_ideal=True, measurement_mode="AC_RMS", window_duration=0.05)
    r_load = Resistor("R1", "vload", "0", resistance=1000.0)
    gnd = Ground("GND", "0")
    c.add(src)
    c.add(am)
    c.add(vm)
    c.add(r_load)
    c.add(gnd)

    sess = SimulationSession(c)
    sess.start()
    # Step 5 full 100 Hz cycles (50 ms) at 250 us/step -> 200 steps
    frame = sess.step_chunk(steps=200, dt=2.5e-4)

    expected_vrms = 10.0 / math.sqrt(2.0)
    expected_irms = 0.01 / math.sqrt(2.0)
    assert vm.reading == pytest.approx(expected_vrms, rel=0.03)
    assert am.reading == pytest.approx(expected_irms, rel=0.03)
    assert "Vrms" in vm.format_reading()
    assert "Arms" in am.format_reading()
    assert frame.components["VM1"].voltage == pytest.approx(expected_vrms, rel=0.03)
    assert frame.components["A1"].current == pytest.approx(expected_irms, rel=0.03)
    assert "Vrms" in frame.components["VM1"].display_value
    assert "Arms" in frame.components["A1"].display_value

    # Toggle via QuickValueDialog
    dlg = QuickValueDialog(vm)
    try:
        idx_dc = dlg.meter_mode_combo.findData("DC")
        assert idx_dc >= 0
        dlg.meter_mode_combo.setCurrentIndex(idx_dc)
        assert dlg._apply_current_inputs() is True
        assert vm.measurement_mode == "DC"
        assert vm.mode == MeterMode.INSTANTANEOUS

        idx_rms = dlg.meter_mode_combo.findData("AC_RMS")
        dlg.meter_mode_combo.setCurrentIndex(idx_rms)
        assert dlg._apply_current_inputs() is True
        assert vm.measurement_mode == "AC_RMS"
        assert vm.mode == MeterMode.RMS
    finally:
        dlg.close()

    # Persist through .circ save/load
    doc = CircuitDocument(
        placed=[
            PlacedComponentSpec(component=vm, center_px=(0.0, 0.0)),
            PlacedComponentSpec(component=am, center_px=(100.0, 0.0)),
        ]
    )
    out_file = tmp_path / "rms_meters.circ"
    save_document(out_file, doc)
    reloaded = load_document(out_file)
    vm_r = reloaded.placed[0].component
    am_r = reloaded.placed[1].component
    assert isinstance(vm_r, Voltmeter) and vm_r.measurement_mode == "AC_RMS" and vm_r.mode == MeterMode.RMS
    assert isinstance(am_r, Ammeter) and am_r.measurement_mode == "AC_RMS" and am_r.mode == MeterMode.RMS


def test_m4_logic_probe_net_logic_coloring_and_truth_table_generator(
    qapp: QApplication, tmp_path
) -> None:
    from PySide6.QtCore import QPointF
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings
    from circuitsim.canvas.scene import CircuitScene
    from circuitsim.canvas.symbol_renderer import SymbolLibrary
    from circuitsim.components.ground import Ground
    from circuitsim.components.logic_gate import LogicGate
    from circuitsim.components.logic_probe import LogicProbe
    from circuitsim.components.sources import DCVoltageSource
    from circuitsim.dialogs.truth_table_dialog import TruthTableDialog
    from circuitsim.engine.netlist import Circuit
    from circuitsim.engine.solver import MNASolver
    from circuitsim.engine.truth_table import generate_truth_table
    from circuitsim.io.circ_serializer import (
        CircuitDocument,
        PlacedComponentSpec,
        load_document,
        save_document,
    )
    from circuitsim.wiring.overlay import map_logic_voltage_to_rgb
    from tests.ui_flow_helpers import terminal_scene_point

    # 1. LogicProbe thresholds & family presets + DC simulation
    assert map_logic_voltage_to_rgb(5.0) == (34, 197, 94)
    assert map_logic_voltage_to_rgb(0.2) == (56, 189, 248)
    assert map_logic_voltage_to_rgb(2.5) == (245, 158, 11)

    lp_ttl = LogicProbe("LP_TTL", "n_test", family="5V_TTL")
    assert lp_ttl.v_il == pytest.approx(0.8)
    assert lp_ttl.v_ih == pytest.approx(2.0)
    assert lp_ttl.evaluate_voltage(2.5) == "H"
    assert lp_ttl.evaluate_voltage(1.2) == "?"
    assert lp_ttl.evaluate_voltage(0.4) == "L"

    # 2. NAND + NOT -> AND gate circuit with LogicProbe on output
    c = Circuit()
    va = DCVoltageSource("VA", "in_a", "0", voltage=5.0)
    vb = DCVoltageSource("VB", "in_b", "0", voltage=5.0)
    u_nand = LogicGate("U1", "nand", ("in_a", "in_b"), "n_nand")
    u_not = LogicGate("U2", "not", "n_nand", "n_out")
    lp = LogicProbe("LP1", "n_out")
    gnd = Ground("GND", "0")
    for comp in (va, vb, u_nand, u_not, lp, gnd):
        c.add(comp)

    c.validate_topology()
    dc_res = MNASolver(c).solve_dc()
    assert dc_res.node_voltages["n_out"] > 4.0
    assert lp.logic_state == "H"

    tt = generate_truth_table(c)
    assert tt.input_names == ("VA", "VB")
    assert tt.output_names == ("LP1",)
    assert tt.as_mapping() == {
        (0, 0): ("0",),
        (0, 1): ("0",),
        (1, 0): ("0",),
        (1, 1): ("1",),
    }
    md = tt.to_markdown()
    csv_txt = tt.to_csv()
    assert "| VA | VB | LP1 |" in md
    assert "1,1,1" in csv_txt

    # 3. TruthTableDialog UI check
    dlg = TruthTableDialog(lambda: c)
    try:
        res_ui = dlg.run_generation()
        assert res_ui is not None
        assert dlg.table.rowCount() == 4
        assert dlg.table.columnCount() == 3
    finally:
        dlg.close()

    # 4. Net logic coloring on CircuitScene + View menu action check
    lib = SymbolLibrary()
    scene = CircuitScene(lib)
    it_va = scene.place_new_component("dc_voltage", QPointF(-200.0, -60.0))
    it_u1 = scene.place_new_component("gate_not", QPointF(-40.0, -60.0))
    it_lp = scene.place_new_component("logic_probe", QPointF(120.0, -60.0))
    it_gnd = scene.place_new_component("ground", QPointF(-200.0, 40.0))

    it_va.component.dc_voltage = 5.0  # type: ignore[attr-defined]
    w_in = scene.add_routed_wire(
        terminal_scene_point(it_va, "pos"),
        terminal_scene_point(it_u1, "in"),
    )
    scene.add_routed_wire(
        terminal_scene_point(it_va, "neg"),
        terminal_scene_point(it_gnd, "gnd"),
    )
    w_out = scene.add_routed_wire(
        terminal_scene_point(it_u1, "out"),
        terminal_scene_point(it_lp, "in"),
    )

    conn = scene.extract_connectivity_result()
    conn.circuit.validate_topology()
    res_scene = MNASolver(conn.circuit).solve_dc()
    scene.apply_solve_result(res_scene)
    scene.set_net_logic_coloring_enabled(True)

    assert it_lp.component.logic_state == "L"  # type: ignore[attr-defined]
    assert w_in.overlay_rgb == (34, 197, 94)  # 5 V High -> green
    assert w_out.overlay_rgb == (56, 189, 248)  # 0 V Low -> blue

    # 5. .circ round-trip for LogicProbe
    doc = CircuitDocument(
        placed=[PlacedComponentSpec(component=lp_ttl, center_px=(0.0, 0.0))]
    )
    lp_file = tmp_path / "logic_probe.circ"
    save_document(lp_file, doc)
    loaded = load_document(lp_file)
    lp_loaded = loaded.placed[0].component
    assert isinstance(lp_loaded, LogicProbe)
    assert lp_loaded.family == "5V_TTL"

    # 6. MainWindow View & Simulation menu hooks exist
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "m4_settings"))
    try:
        assert hasattr(win, "act_net_logic_colors")
        assert hasattr(win, "act_truth_table")
    finally:
        win.circuit_scene.undo_stack.setClean()
        win._switch_modified = False
        win.close()


def test_m5_potentiometer_bracket_tuning_and_space_quick_place(
    qapp: QApplication, tmp_path
) -> None:
    from PySide6.QtCore import QPointF, QTimer, Qt
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings
    from circuitsim.components.potentiometer import Potentiometer
    from circuitsim.dialogs.quick_place_dialog import QuickPlaceDialog
    from circuitsim.engine.solver import MNASolver
    from tests.ui_flow_helpers import press_key, terminal_scene_point

    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "m5_settings"))
    scene = win.circuit_scene
    view = win.circuit_view
    try:
        win.show()
        qapp.processEvents()

        # Build 10 V + Potentiometer voltage divider (node_a = +10V, node_b = GND)
        it_v = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0))
        it_g = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        it_pot = scene.place_new_component("potentiometer", QPointF(0.0, 0.0))
        it_v.component.dc_voltage = 10.0  # type: ignore[attr-defined]

        scene.add_routed_wire(terminal_scene_point(it_v, "pos"), terminal_scene_point(it_pot, "a"))
        scene.add_routed_wire(terminal_scene_point(it_v, "neg"), terminal_scene_point(it_g, "gnd"))
        scene.add_routed_wire(terminal_scene_point(it_pot, "b"), terminal_scene_point(it_g, "gnd"))

        pot = it_pot.component
        assert isinstance(pot, Potentiometer)
        assert pot.alpha == pytest.approx(0.5)

        scene.clearSelection()
        it_pot.setSelected(True)

        # Press ']' twice (+0.05 each -> 0.60), then Shift+'[' (-0.01 -> 0.59)
        press_key(view, Qt.Key.Key_BracketRight)
        assert pot.alpha == pytest.approx(0.55)
        assert "55%" in win.statusBar().currentMessage()

        press_key(view, Qt.Key.Key_BracketRight)
        assert pot.alpha == pytest.approx(0.60)

        press_key(view, Qt.Key.Key_BraceLeft, modifiers=Qt.KeyboardModifier.ShiftModifier)
        assert pot.alpha == pytest.approx(0.59)

        # Wiper at alpha=0.59 from GND(b) -> A(+10V): V_w = 10.0 * 0.59 = 5.90 V
        conn = scene.extract_connectivity_result()
        dc_res = MNASolver(conn.circuit).solve_dc()
        w_node = conn.component_nodes[pot.name]["w"]
        v_wiper = float(dc_res.node_voltages[w_node])
        assert v_wiper == pytest.approx(5.90, abs=0.05)

        # Spacebar triggers QuickPlaceDialog (non-blocking) and places chosen component near cursor
        view._last_scene_pos = QPointF(220.0, -40.0)
        count_before = len(scene.placed_items)

        press_key(view, Qt.Key.Key_Space)
        dlg = view.quick_place_dialog
        assert isinstance(dlg, QuickPlaceDialog)
        dlg.search_input.setText("nand")
        qapp.processEvents()
        dlg.accept()
        qapp.processEvents()

        assert len(scene.placed_items) == count_before + 1
        placed_types = [it.component.component_type for it in scene.placed_items]
        assert "gate_nand" in placed_types
    finally:
        win.circuit_scene.undo_stack.setClean()
        win._switch_modified = False
        win.close()


def test_m6_custom_parts_library_preset_macro_export_import_and_palette(
    qapp: QApplication, tmp_path
) -> None:
    from PySide6.QtCore import QPointF, Qt
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings
    from circuitsim.components.custom_macro import CustomMacroBlock, package_subcircuit_macro
    from circuitsim.components.resistor import Resistor
    from circuitsim.dialogs.custom_parts_dialog import ManageCustomPartsDialog
    from circuitsim.engine.solver import MNASolver
    from circuitsim.io.circ_serializer import (
        CircuitDocument,
        PlacedComponentSpec,
        load_document,
        save_document,
    )
    from circuitsim.io.user_library import (
        UserLibraryStore,
        create_macro_part_entry,
        create_preset_from_component,
    )
    from tests.ui_flow_helpers import terminal_scene_point

    lib_path = tmp_path / "user_library.json"
    store = UserLibraryStore(lib_path)

    # 1. Save a Preset (10 kOhm Resistor) and a Macro (1:1 Voltage Divider)
    r_template = Resistor("R_tpl", "a", "b", resistance=10_000.0)
    preset_entry = create_preset_from_component(
        r_template, name="Pullup 10k", description="Standard I2C pull-up"
    )
    store.add_or_update_part(preset_entry)

    div_r1 = Resistor("R_top", "IN", "OUT", resistance=1000.0)
    div_r2 = Resistor("R_bot", "OUT", "GND", resistance=1000.0)
    div_macro_def = package_subcircuit_macro("Divider Macro", ("IN", "OUT", "GND"), (div_r1, div_r2))
    macro_entry = create_macro_part_entry(div_macro_def, description="50% 1k/1k divider")
    store.add_or_update_part(macro_entry)

    # Reload from disk in fresh store
    reloaded = UserLibraryStore(lib_path)
    assert len(reloaded.list_parts()) == 2
    assert reloaded.get_part(preset_entry.part_id) is not None
    assert reloaded.get_part(macro_entry.part_id) is not None

    # Export / Import round-trip
    export_file = tmp_path / "exported_parts.json"
    reloaded.export_library(export_file)
    store2 = UserLibraryStore(tmp_path / "imported_library.json")
    imported = store2.import_library(export_file)
    assert len(imported) == 2

    # 2. Palette, Manage Dialog, Scene Placement & MNA Simulation in MainWindow
    win = MainWindow(
        settings=AppSettings(data_dir=tmp_path / "m6_settings"),
        user_library=store,
    )
    try:
        top_cat = win.palette_widget.tree.topLevelItem(0)
        assert top_cat is not None
        assert top_cat.text(0) == "My Custom Parts"
        names = [top_cat.child(i).text(0) for i in range(top_cat.childCount())]
        assert "Pullup 10k" in names
        assert "Divider Macro" in names

        mgr = ManageCustomPartsDialog(
            store,
            on_library_changed=win.palette_widget.refresh_custom_parts,
            parent=win,
        )
        try:
            assert mgr.parts_list.count() == 2
        finally:
            mgr.close()

        scene = win.circuit_scene
        # Place custom preset
        it_preset = scene.place_new_component(
            f"custom:{preset_entry.part_id}", QPointF(-240.0, -120.0)
        )
        assert it_preset is not None
        assert isinstance(it_preset.component, Resistor)
        assert it_preset.component.resistance == pytest.approx(10_000.0)

        # Place custom macro + DC source + Ground and verify MNA DC solution
        it_v = scene.place_new_component("dc_voltage", QPointF(-160.0, 40.0))
        it_g = scene.place_new_component("ground", QPointF(-160.0, 160.0))
        it_mac = scene.place_new_component(
            f"custom:{macro_entry.part_id}", QPointF(40.0, 40.0)
        )
        assert it_v is not None and it_g is not None and it_mac is not None
        assert isinstance(it_mac.component, CustomMacroBlock)
        it_v.component.dc_voltage = 10.0  # type: ignore[attr-defined]

        scene.add_routed_wire(terminal_scene_point(it_v, "pos"), terminal_scene_point(it_mac, "IN"))
        scene.add_routed_wire(terminal_scene_point(it_v, "neg"), terminal_scene_point(it_g, "gnd"))
        scene.add_routed_wire(terminal_scene_point(it_mac, "GND"), terminal_scene_point(it_g, "gnd"))

        conn = scene.extract_connectivity_result()
        conn.circuit.validate_topology()
        dc_res = MNASolver(conn.circuit).solve_dc()
        out_net = conn.component_nodes[it_mac.component.name]["OUT"]
        assert float(dc_res.node_voltages[out_net]) == pytest.approx(5.0, abs=1e-3)

        # Verify CustomMacroBlock round-trips through .circ save/load
        circ_path = tmp_path / "macro_roundtrip.circ"
        save_document(
            circ_path,
            CircuitDocument(placed=[PlacedComponentSpec(component=it_mac.component, center_px=(40.0, 40.0))]),
        )
        loaded_doc = load_document(circ_path)
        mac_loaded = loaded_doc.placed[0].component
        assert isinstance(mac_loaded, CustomMacroBlock)
        assert mac_loaded.macro_def.macro_name == "Divider Macro"
        assert tuple(mac_loaded.macro_def.pins) == ("IN", "OUT", "GND")
    finally:
        win.circuit_scene.undo_stack.setClean()
        win._switch_modified = False
        win.close()

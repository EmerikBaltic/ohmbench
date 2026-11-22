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

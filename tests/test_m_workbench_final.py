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

"""Tests for Item F6: Qt-free AC analysis (V(jw), Z_in, Gamma) & Benchtop Instruments UI."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import numpy as np

from circuitsim.components import (
    ACVoltageSource,
    Capacitor,
    DCVoltageSource,
    Diode,
    Ground,
    Inductor,
    Resistor,
)
from circuitsim.engine import ACAnalysisResult, Circuit, MNASolver, run_ac_sweep


def test_f6_ac_analysis_module_is_qt_free() -> None:
    """Verify `circuitsim.engine.ac_analysis` imports no Qt modules."""
    ac_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "circuitsim"
        / "engine"
        / "ac_analysis.py"
    )
    tree = ast.parse(ac_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("PySide6", "PyQt")), alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("PySide6", "PyQt")), node.module


def test_f6_rc_lowpass_bode_magnitude_db_and_phase_deg() -> None:
    """Verify exact RC low-pass V(jw), dB, and phase deg at 10 Hz, f_c=1 kHz, 10 kHz."""
    f_c = 1000.0
    r_val = 1000.0
    c_val = 1.0 / (2.0 * math.pi * r_val * f_c)  # 159.1549 nF

    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(ACVoltageSource("V1", "vin", "0", amplitude=1.0, frequency=1000.0))
    c.add(Resistor("R1", "vin", "vout", resistance=r_val))
    c.add(Capacitor("C1", "vout", "0", capacitance=c_val, esr=0.0))

    res = run_ac_sweep(c, frequencies_hz=[10.0, 1000.0, 10000.0])
    assert isinstance(res, ACAnalysisResult)

    # Analytical transfer: H(jw) = 1 / (1 + j*(f/f_c))
    # 10 Hz: f/fc = 0.01 -> |H| = 0.99995 (-0.00043 dB), phase = -0.5729 deg
    s0 = res.sample_at("vout", 0)
    assert math.isclose(s0.mag_v, 1.0 / math.sqrt(1.0 + 0.01**2), rel_tol=1e-9)
    assert math.isclose(s0.db, 20.0 * math.log10(1.0 / math.sqrt(1.0 + 0.01**2)), abs_tol=1e-6)
    assert math.isclose(s0.phase_deg, -math.degrees(math.atan(0.01)), abs_tol=1e-6)

    # 1000 Hz (f_c): |H| = 1/sqrt(2) (-3.0103 dB), phase = -45.000 deg
    s1 = res.sample_at("vout", 1)
    assert math.isclose(s1.mag_v, 1.0 / math.sqrt(2.0), rel_tol=1e-9)
    assert math.isclose(s1.db, -3.0102999566, abs_tol=1e-6)
    assert math.isclose(s1.phase_deg, -45.0, abs_tol=1e-6)

    # 10000 Hz (10 * f_c): |H| = 1/sqrt(101) (-20.0432 dB), phase = -84.2894 deg
    s2 = res.sample_at("vout", 2)
    assert math.isclose(s2.mag_v, 1.0 / math.sqrt(101.0), rel_tol=1e-9)
    assert math.isclose(s2.db, -20.0432137, abs_tol=1e-5)
    assert math.isclose(s2.phase_deg, -math.degrees(math.atan(10.0)), abs_tol=1e-5)


def test_f6_series_rlc_resonance_zin_and_gamma() -> None:
    """Verify series RLC resonant Z_in = 50 + j0 ohm and |Gamma| < 0.001 at 31.831 kHz."""
    r_val = 50.0
    l_val = 1e-3
    c_val = 25e-9
    f_0 = 1.0 / (2.0 * math.pi * math.sqrt(l_val * c_val))  # 31830.9886 Hz

    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(ACVoltageSource("V1", "vin", "0", amplitude=1.0, frequency=f_0))
    c.add(Resistor("R1", "vin", "n1", resistance=r_val))
    c.add(Inductor("L1", "n1", "n2", inductance=l_val, dcr=1e-3))
    c.add(Capacitor("C1", "n2", "0", capacitance=c_val, esr=0.0))

    res = run_ac_sweep(c, frequencies_hz=[1000.0, f_0, 1e6], z0_ohm=50.0)
    # At resonance: Z_in ~ 50.001 + j0 ohm, |Gamma| << 0.001
    z_res = complex(res.z_in[1])
    assert math.isclose(z_res.real, 50.001, abs_tol=1e-2)
    assert abs(z_res.imag) < 1e-5
    assert float(res.gamma_mag[1]) < 0.001

    # Far off-resonance (1 kHz): capacitive reactance dominates, high reflection |Gamma| > 0.98
    assert float(res.gamma_mag[0]) > 0.98


def test_f6_ac_small_signal_diode_linearization() -> None:
    """Verify AC sweep linearizes nonlinear Shockley Diode around its DC operating point."""
    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "vin", "0", voltage=5.0))
    c.add(Resistor("R1", "vin", "vout", resistance=1000.0))
    d1 = Diode("D1", "vout", "0")
    c.add(d1)

    res = run_ac_sweep(c, frequencies_hz=[100.0, 1000.0], source_name="V1")
    # Small-signal dynamic resistance r_d = 1 / g_d at DC operating point
    dc_solver = MNASolver(c)
    dc_solver.solve_dc()
    r_d = 1.0 / d1.companion_model(0.0, "dc")[0]
    expected_gain = r_d / (1000.0 + r_d)
    assert math.isclose(res.sample_at("vout", 1).mag_v, expected_gain, rel_tol=1e-6)


def test_f6_benchtop_instruments_panel_scope_bode_and_cursors(qapp, tmp_path: Path) -> None:
    """Verify BenchtopInstrumentsDialog computes V_pp/V_rms/V_avg/freq, cursors, and Bode/Z_in."""
    from circuitsim.app.instruments_panel import BenchtopInstrumentsDialog
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings

    # 1. RC step capture: tau = 1k * 1uF = 1 ms, V_s = 10 V -> V(1 ms)=6.321 V, V(5 ms)=9.933 V
    c_rc = Circuit()
    c_rc.add(Ground("GND", "0"))
    c_rc.add(DCVoltageSource("V1", "vin", "0", voltage=10.0))
    c_rc.add(Resistor("R1", "vin", "vout", resistance=1000.0))
    c_rc.add(Capacitor("C1", "vout", "0", capacitance=1e-6, initial_voltage=0.0))

    dlg = BenchtopInstrumentsDialog(circuit_provider=lambda: c_rc)
    try:
        frame = dlg.capture_transient(
            c_rc, duration_s=6e-3, dt_s=1e-5, channel_nodes={"CH1": "vout", "CH2": "vin"}
        )
        assert frame.traces is not None and "CH1" in frame.traces
        dlg.cursor_t1_spin.setValue(1.0)  # 1 ms
        dlg.cursor_t2_spin.setValue(5.0)  # 5 ms
        cr = dlg.latest_cursor_readout
        assert cr is not None
        assert math.isclose(cr.v1, 10.0 * (1.0 - math.exp(-1.0)), abs_tol=0.02)
        assert math.isclose(cr.v2, 10.0 * (1.0 - math.exp(-5.0)), abs_tol=0.02)

        # 2. Periodic sine capture: amp=5 V, dc_offset=2 V, f=1 kHz -> Vpp=10, Vavg=2, Vrms=sqrt(16.5)
        c_sin = Circuit()
        c_sin.add(Ground("GND", "0"))
        c_sin.add(
            ACVoltageSource(
                "V1", "vsig", "0", amplitude=5.0, frequency=1000.0, dc_offset=2.0
            )
        )
        c_sin.add(Resistor("R1", "vsig", "0", resistance=1000.0))

        frame2 = dlg.capture_transient(
            c_sin, duration_s=5e-3, dt_s=5e-6, channel_nodes={"CH1": "vsig"}
        )
        meas = frame2.traces["CH1"].measurements
        assert math.isclose(meas.vpp, 10.0, rel_tol=1e-2)
        assert math.isclose(meas.vavg, 2.0, abs_tol=0.05)
        assert math.isclose(meas.vrms, math.sqrt(2.0**2 + (5.0 / math.sqrt(2.0)) ** 2), rel_tol=2e-2)
        assert meas.frequency_hz is not None
        assert math.isclose(meas.frequency_hz, 1000.0, rel_tol=2e-2)

        # 3. AC Bode + Z_in tab on series RLC (31.831 kHz resonance)
        c_rlc = Circuit()
        c_rlc.add(Ground("GND", "0"))
        c_rlc.add(ACVoltageSource("V1", "vin", "0", amplitude=1.0, frequency=31831.0))
        c_rlc.add(Resistor("R1", "vin", "n1", resistance=50.0))
        c_rlc.add(Inductor("L1", "n1", "n2", inductance=1e-3, dcr=1e-3))
        c_rlc.add(Capacitor("C1", "n2", "0", capacitance=25e-9, esr=0.0))

        ac_res = dlg.run_ac_analysis(
            c_rlc, probe_node="n2", f_start=1000.0, f_stop=1e6, points_per_decade=40
        )
        dlg.bode_cursor_f_spin.setValue(31831.0)
        assert ac_res is dlg.latest_ac_result
        assert "Gamma" in dlg.bode_readout_label.text()
    finally:
        dlg.close()

    # 4. MainWindow launcher opens BenchtopInstrumentsDialog cleanly
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        inst = win.show_instruments_panel()
        assert inst.isVisible()
    finally:
        win.close()

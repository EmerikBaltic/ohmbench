"""Tests for Item F5: solver robustness, convergence recovery, and Shockley Diode."""

from __future__ import annotations

import math
import numpy as np
import pytest

from circuitsim.components import (
    ACVoltageSource,
    Capacitor,
    DCVoltageSource,
    Diode,
    Ground,
    Inductor,
    KNOWN_COMPONENT_TYPES,
    Resistor,
    create_default_component_model,
)
from circuitsim.engine import Circuit, MNASolver, VoltageSourceLoopError
from circuitsim.io.circ_serializer import CircuitDocument, PlacedComponentSpec, dumps_document, loads_document


def _build_clipper_circuit(amplitude: float) -> tuple[Circuit, Resistor, Diode, Diode]:
    circuit = Circuit()
    v1 = ACVoltageSource("V1", "vin", "0", amplitude=amplitude, frequency=1000.0)
    r1 = Resistor("R1", "vin", "vout", resistance=1000.0)
    d1 = Diode("D1", "vout", "0")
    d2 = Diode("D2", "0", "vout")
    gnd = Ground("GND", "0")
    for c in (v1, r1, d1, d2, gnd):
        circuit.add(c)
    return circuit, r1, d1, d2


def test_f5_shockley_diode_iv_and_reverse_breakdown() -> None:
    """Verify exact Shockley forward equation with N, I_s, R_s and reverse V_br knee."""
    d = Diode(
        "D1",
        "a",
        "k",
        is_sat=2.52e-9,
        n=1.752,
        rs=0.568,
        bv=10.0,
        ibv=1e-4,
    )
    i_target = 10e-3
    vd_exact = d.n * d.vt * math.log(i_target / d.is_sat + 1.0)
    # Include g_min_leak term consistently when computing v_ak
    vak_exact = vd_exact + (d.junction_current(vd_exact)) * d.rs
    assert math.isclose(d.current_at_voltage(vak_exact), i_target, rel_tol=1e-8)

    # Reverse breakdown check: V_supply = 15 V through 1k into cathode with bv = 5.1 V.
    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "src", "0", voltage=15.0))
    c.add(Resistor("R1", "src", "k", resistance=1000.0))
    c.add(
        Diode(
            "DZ",
            "0",
            "k",
            is_sat=1e-12,
            n=1.0,
            rs=0.5,
            bv=5.1,
            ibv=1e-3,
        )
    )
    res = MNASolver(c).solve_dc()
    v_z = res.node_voltages["k"]
    i_r1 = (15.0 - v_z) / 1000.0
    # At ~9.8 mA (~10x IBV), reverse knee drop is bv + N*V_T*ln(i/ibv) + i*rs ~ 5.16 V.
    assert 5.10 < v_z < 5.30
    assert math.isclose(res.branch_currents["DZ"], -i_r1, rel_tol=1e-7)


def test_f5_diode_clipper_10vpp_vs_01vpp_linear_pass() -> None:
    """Verify symmetric 10 Vpp diode clipping (~0.65 V) and 0.1 Vpp linear pass."""
    c_clip, _, _, _ = _build_clipper_circuit(amplitude=5.0)
    res_clip = MNASolver(c_clip).solve_transient(duration=2e-3, dt=1e-5)
    v_out = res_clip.node_voltages["vout"]
    peak_pos = float(np.max(v_out))
    peak_neg = float(np.min(v_out))
    # Analytical peak current ~ (5.0 - 0.653)/1000 = 4.35 mA -> V_d ~ 0.653 V.
    assert 0.63 <= peak_pos <= 0.72
    assert -0.72 <= peak_neg <= -0.63
    assert math.isclose(peak_pos, -peak_neg, rel_tol=5e-3)

    # Small-signal 0.1 Vpp (50 mV amplitude): diode draws ~nA -> passes linearly.
    c_lin, _, _, _ = _build_clipper_circuit(amplitude=0.05)
    res_lin = MNASolver(c_lin).solve_transient(duration=2e-3, dt=1e-5)
    v_lin = res_lin.node_voltages["vout"]
    assert math.isclose(float(np.max(v_lin)), 0.05, rel_tol=5e-3)
    assert math.isclose(float(np.min(v_lin)), -0.05, rel_tol=5e-3)


def test_f5_junction_delta_v_damping_and_stiff_dc_convergence() -> None:
    """Verify pnjlim step damping prevents exponential overflow on stiff drives."""
    d = Diode("D1", "a", "0", is_sat=1e-14, n=1.0, rs=0.5, bv=100.0)
    v_lim, limited = d.limit_junction_voltage(5.0, 0.0)
    assert limited is True
    assert 0.05 < v_lim < 0.35

    v_small, limited_small = d.limit_junction_voltage(0.01, 0.009)
    assert limited_small is False
    assert math.isclose(v_small, 0.01, rel_tol=1e-12)

    # Stiff 100 V forward drive across 100 ohm -> exp(100/Vt) would overflow naive NR.
    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "vin", "0", voltage=100.0))
    c.add(Resistor("R1", "vin", "a", resistance=100.0))
    c.add(d)

    solver = MNASolver(c)
    res = solver.solve_dc()
    v_a = res.node_voltages["a"]
    i_d = res.branch_currents["D1"]
    assert solver.last_convergence_method in ("newton_damped", "direct")
    assert math.isclose(i_d, (100.0 - v_a) / 100.0, rel_tol=1e-8)
    assert math.isclose(d.current_at_voltage(v_a), i_d, rel_tol=1e-8)


def test_f5_gmin_and_source_stepping_convergence_recovery() -> None:
    """Verify Gmin stepping and Source stepping both recover multi-diode networks."""
    def make_stack() -> Circuit:
        c = Circuit()
        c.add(Ground("GND", "0"))
        c.add(DCVoltageSource("V1", "n0", "0", voltage=30.0))
        c.add(Resistor("R1", "n0", "n1", resistance=500.0))
        c.add(Diode("D1", "n1", "n2", is_sat=1e-14, n=1.0, rs=0.2))
        c.add(Diode("D2", "n2", "n3", is_sat=1e-14, n=1.0, rs=0.2))
        c.add(Diode("D3", "n3", "n4", is_sat=1e-14, n=1.0, rs=0.2))
        c.add(Diode("D4", "n4", "0", is_sat=1e-14, n=1.0, rs=0.2))
        return c

    res_direct = MNASolver(make_stack()).solve_dc()

    s_gmin = MNASolver(make_stack())
    res_gmin = s_gmin.solve_dc(recovery_mode="gmin")
    assert s_gmin.last_convergence_method == "gmin_stepping"
    assert math.isclose(res_gmin.node_voltages["n1"], res_direct.node_voltages["n1"], rel_tol=1e-7)

    s_src = MNASolver(make_stack())
    res_src = s_src.solve_dc(recovery_mode="source")
    assert s_src.last_convergence_method == "source_stepping"
    assert math.isclose(res_src.node_voltages["n1"], res_direct.node_voltages["n1"], rel_tol=1e-7)

    # Automatic fallback when direct Newton iteration cap is tight (direct_max_iters=2)
    s_auto = MNASolver(make_stack())
    res_auto = s_auto.solve_dc(direct_max_iters=2)
    assert s_auto.last_convergence_method in ("gmin_stepping", "source_stepping")
    assert math.isclose(res_auto.node_voltages["n1"], res_direct.node_voltages["n1"], rel_tol=1e-7)


def test_f5_ideal_loop_regularization_and_adaptive_transient_recovery() -> None:
    """Verify ideal loop regularization and steep transient switching recovery."""
    c_loop = Circuit()
    c_loop.add(Ground("GND", "0"))
    c_loop.add(DCVoltageSource("V1", "a", "0", voltage=5.0, r_internal=0.0))
    c_loop.add(DCVoltageSource("V2", "a", "0", voltage=3.0, r_internal=0.0))

    # Default strict topology validation preserved for existing behavior
    with pytest.raises(VoltageSourceLoopError):
        MNASolver(c_loop).solve_dc()

    # With ideal loop regularization enabled: solves to symmetric 4.0 V with warning
    s_reg = MNASolver(c_loop, regularize_ideal_loops=True)
    res_reg = s_reg.solve_dc()
    assert math.isclose(res_reg.node_voltages["a"], 4.0, abs_tol=1e-4)
    assert any(w.code == "ideal_loop_regularized" for w in res_reg.warnings)

    # Parallel ideal capacitor loop with zero ESR at t=0+ regularizes cleanly
    c_cap = Circuit()
    c_cap.add(Ground("GND", "0"))
    c_cap.add(DCVoltageSource("V1", "a", "0", voltage=5.0, r_internal=0.0))
    c_cap.add(Capacitor("C1", "a", "0", capacitance=1e-6, esr=0.0))
    c_cap.add(Capacitor("C2", "a", "0", capacitance=2e-6, esr=0.0))
    res_cap = MNASolver(c_cap).solve_transient(duration=1e-4, dt=1e-5)
    assert math.isclose(float(res_cap.node_voltages["a"][-1]), 5.0, rel_tol=1e-4)

    # Steep +/-20 V square wave into sharp clamping diodes + capacitor completes cleanly
    c_sw = Circuit()
    c_sw.add(Ground("GND", "0"))
    c_sw.add(ACVoltageSource("V1", "in", "0", amplitude=20.0, frequency=2000.0, waveform="square"))
    c_sw.add(Resistor("R1", "in", "out", resistance=10.0))
    c_sw.add(Capacitor("C1", "out", "0", capacitance=1e-8))
    c_sw.add(Diode("D1", "out", "0", is_sat=1e-14, n=1.0, rs=0.1))
    c_sw.add(Diode("D2", "0", "out", is_sat=1e-14, n=1.0, rs=0.1))
    res_sw = MNASolver(c_sw).solve_transient(duration=1e-3, dt=2.5e-5)
    assert np.all(np.isfinite(res_sw.node_voltages["out"]))


def test_f5_deterministic_symmetric_oscillator_startup_perturbation() -> None:
    """Verify zero-IC symmetric reactive oscillator starts deterministically via +1 uV."""
    def run_tank() -> tuple[MNASolver, np.ndarray]:
        c = Circuit()
        c.add(Ground("GND", "0"))
        # Symmetric 10 mH inductor + dual 1 uF capacitors starting at exact 0 V / 0 A.
        c.add(Inductor("L1", "n1", "0", inductance=10e-3, dcr=1e-3))
        c.add(Capacitor("C1", "n1", "0", capacitance=1e-6, initial_voltage=0.0))
        c.add(Capacitor("C2", "n1", "0", capacitance=1e-6, initial_voltage=0.0))
        solver = MNASolver(c)
        # Period T = 2*pi*sqrt(10e-3 * 2e-6) = 888.58 us
        res = solver.solve_transient(duration=1e-3, dt=2e-6)
        return solver, res.node_voltages["n1"]

    s1, wave1 = run_tank()
    s2, wave2 = run_tank()
    assert s1._startup_perturbation_applied is True
    assert float(np.max(np.abs(wave1))) > 1e-7
    assert np.array_equal(wave1, wave2)


def test_f5_diode_catalog_and_circ_round_trip(qapp) -> None:
    """Verify Diode registers in KNOWN_COMPONENT_TYPES, factory, and .circ serializer."""
    assert "diode" in KNOWN_COMPONENT_TYPES
    d = create_default_component_model("diode", "D1")
    assert isinstance(d, Diode)
    assert d.get_editable_parameters()

    doc = CircuitDocument(
        placed=[
            PlacedComponentSpec(
                component=Diode("D1", "a", "k", is_sat=5e-9, n=1.8, rs=0.4, bv=75.0),
                center_px=(120.0, 120.0),
            )
        ]
    )
    payload = dumps_document(doc)
    loaded = loads_document(payload)
    d_loaded = loaded.placed[0].component
    assert isinstance(d_loaded, Diode)
    assert math.isclose(d_loaded.is_sat, 5e-9, rel_tol=1e-9)
    assert math.isclose(d_loaded.n, 1.8, rel_tol=1e-9)
    assert math.isclose(d_loaded.rs, 0.4, rel_tol=1e-9)
    assert math.isclose(d_loaded.bv, 75.0, rel_tol=1e-9)

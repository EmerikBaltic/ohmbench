"""Unit, physics, timing, and diagnostic tests for digital logic gates (pure Python engine)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_module_has_no_qt_imports
from circuitsim.components import (
    GATE_COMPONENT_TYPES,
    LOGIC_FAMILY_PRESETS,
    ACVoltageSource,
    Capacitor,
    DCCurrentSource,
    DCVoltageSource,
    Ground,
    KNOWN_COMPONENT_TYPES,
    LogicGate,
    Resistor,
    create_default_component_model,
)
from circuitsim.engine import (
    Circuit,
    IntegrationMethod,
    MNASolver,
    SimulationSession,
    SimulationState,
    ZeroDelayLoopError,
)


def _rising_midpoint_times(
    times: np.ndarray, values: np.ndarray, midpoint: float
) -> list[float]:
    """Interpolate strictly rising threshold crossing instants from a transient trace."""
    crossings: list[float] = []
    for i in range(1, len(times)):
        v0 = float(values[i - 1])
        v1 = float(values[i])
        if v0 < midpoint <= v1 and v1 > v0:
            frac = (midpoint - v0) / (v1 - v0)
            crossings.append(float(times[i - 1] + frac * (times[i] - times[i - 1])))
    return crossings


def _falling_midpoint_times(
    times: np.ndarray, values: np.ndarray, midpoint: float
) -> list[float]:
    """Interpolate strictly falling threshold crossing instants from a transient trace."""
    crossings: list[float] = []
    for i in range(1, len(times)):
        v0 = float(values[i - 1])
        v1 = float(values[i])
        if v0 > midpoint >= v1 and v0 > v1:
            frac = (v0 - midpoint) / (v0 - v1)
            crossings.append(float(times[i - 1] + frac * (times[i] - times[i - 1])))
    return crossings


def test_logic_gate_modules_have_no_qt_imports() -> None:
    """Engine and logic-gate component modules must stay completely free of Qt imports."""
    assert_module_has_no_qt_imports(
        "circuitsim.components.logic_gate",
        "circuitsim.components.factory",
        "circuitsim.engine.errors",
        "circuitsim.engine.netlist",
        "circuitsim.engine.solver",
        "circuitsim.engine.simulation_session",
    )


def test_factory_creates_all_seven_logic_gate_types() -> None:
    """All 7 IEC logic gates must register in KNOWN_COMPONENT_TYPES and construct cleanly."""
    for comp_id in GATE_COMPONENT_TYPES:
        assert comp_id in KNOWN_COMPONENT_TYPES
        gate = create_default_component_model(comp_id, f"U_{comp_id}")
        assert isinstance(gate, LogicGate)
        assert gate.component_type == comp_id
        expected_pins = 2 if comp_id == "gate_not" else 3
        assert len(gate.nodes) == expected_pins

    sample = create_default_component_model("gate_nand", "U1")
    assert isinstance(sample, LogicGate)
    assert sample.set_parameter_value("t_pd", "15n") == pytest.approx(15e-9)
    assert sample.set_parameter_value("r_out", "200") == pytest.approx(200.0)


@pytest.mark.parametrize(
    ("family", "v_low_in", "v_high_in"),
    [
        ("5V_CMOS", 0.5, 4.5),
        ("5V_TTL", 0.4, 3.0),
        ("3V3_LVCMOS", 0.3, 3.0),
    ],
)
@pytest.mark.parametrize(
    ("operation", "truth_map"),
    [
        ("and", {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}),
        ("or", {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
        ("nand", {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}),
        ("nor", {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}),
        ("xor", {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}),
        ("xnor", {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}),
    ],
)
def test_binary_gate_truth_tables_across_all_families(
    family: str,
    v_low_in: float,
    v_high_in: float,
    operation: str,
    truth_map: dict[tuple[int, int], int],
) -> None:
    """Every 2-input gate matches its hand-written truth table across all 3 families."""
    preset = LOGIC_FAMILY_PRESETS[family]
    # Include a 100 kOhm load resistor so R_out = 100 Ohm loading follows:
    # V_out = V_drive * R_load / (R_out + R_load)
    r_out = 100.0
    r_load = 100_000.0
    divider = r_load / (r_out + r_load)

    for (a_bit, b_bit), expected_out_bit in truth_map.items():
        circuit = Circuit()
        circuit.add(Ground("GND1", "0"))
        circuit.add(
            DCVoltageSource("VA", "na", "0", voltage=v_high_in if a_bit else v_low_in)
        )
        circuit.add(
            DCVoltageSource("VB", "nb", "0", voltage=v_high_in if b_bit else v_low_in)
        )
        circuit.add(LogicGate("U1", operation, ("na", "nb"), "nout", family=family))
        circuit.add(Resistor("RL", "nout", "0", resistance=r_load))

        res = MNASolver(circuit).solve_dc()
        expected_drive = preset["v_oh"] if expected_out_bit else preset["v_ol"]
        assert res.node_voltages["nout"] == pytest.approx(expected_drive * divider, abs=1e-4)


@pytest.mark.parametrize("family", ["5V_CMOS", "5V_TTL", "3V3_LVCMOS"])
def test_not_gate_truth_table_across_all_families(family: str) -> None:
    """NOT gate inverts Low/High inputs accurately across all 3 logic families."""
    preset = LOGIC_FAMILY_PRESETS[family]
    for in_bit in (0, 1):
        circuit = Circuit()
        circuit.add(Ground("GND1", "0"))
        circuit.add(
            DCVoltageSource(
                "VIN",
                "nin",
                "0",
                voltage=(preset["v_oh"] if in_bit else preset["v_ol"]),
            )
        )
        circuit.add(LogicGate("U1", "not", "nin", "nout", family=family))

        res = MNASolver(circuit).solve_dc()
        expected_v = preset["v_ol"] if in_bit else preset["v_oh"]
        assert res.node_voltages["nout"] == pytest.approx(expected_v, abs=1e-6)


def test_half_adder_combinational_multi_gate_network() -> None:
    """Four-gate structural half-adder computes exact Sum = A XOR B and Carry = A & B."""
    preset = LOGIC_FAMILY_PRESETS["5V_CMOS"]
    # Structural half-adder:
    #   nand1 = NAND(A, B)
    #   carry = NOT(nand1)          = A & B
    #   or_ab = OR(A, B)
    #   sum   = AND(or_ab, nand1)   = (A | B) & ~(A & B) = A XOR B
    for a_bit in (0, 1):
        for b_bit in (0, 1):
            circuit = Circuit()
            circuit.add(Ground("GND1", "0"))
            circuit.add(
                DCVoltageSource("VA", "a", "0", voltage=5.0 if a_bit else 0.0)
            )
            circuit.add(
                DCVoltageSource("VB", "b", "0", voltage=5.0 if b_bit else 0.0)
            )
            # Deliberately add downstream gates first to confirm reverse-order DC convergence.
            circuit.add(LogicGate("U_AND", "and", ("n_or", "n_nand"), "sum"))
            circuit.add(LogicGate("U_INV", "not", "n_nand", "carry"))
            circuit.add(LogicGate("U_OR", "or", ("a", "b"), "n_or"))
            circuit.add(LogicGate("U_NAND", "nand", ("a", "b"), "n_nand"))

            res = MNASolver(circuit).solve_dc()
            expected_sum = a_bit ^ b_bit
            expected_carry = a_bit & b_bit
            assert res.node_voltages["sum"] == pytest.approx(
                preset["v_oh"] if expected_sum else preset["v_ol"], abs=1e-5
            )
            assert res.node_voltages["carry"] == pytest.approx(
                preset["v_oh"] if expected_carry else preset["v_ol"], abs=1e-5
            )


def test_propagation_delay_ramp_and_lu_reuse() -> None:
    """Gate delays output transition by `t_pd`, ramps over `t_edge`, and reuses LU."""
    t_step = 50e-9
    t_pd = 10e-9
    t_edge = 2e-9
    dt = 0.5e-9

    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    # PULSE with frequency=10 MHz (period=100 ns), duty=0.5, phase=180 deg:
    # t in [0, 50 ns) => 0 V, t >= 50 ns => 5 V.
    circuit.add(
        ACVoltageSource(
            "VPULSE",
            "nin",
            "0",
            waveform="pulse",
            amplitude=5.0,
            frequency=10e6,
            duty_cycle=0.5,
            phase_deg=180.0,
        )
    )
    circuit.add(
        LogicGate("U1", "not", "nin", "nout", family="5V_CMOS", t_pd=t_pd, t_edge=t_edge)
    )

    res = MNASolver(circuit).solve_transient(duration=80e-9, dt=dt)
    v_ol = LOGIC_FAMILY_PRESETS["5V_CMOS"]["v_ol"]
    v_oh = LOGIC_FAMILY_PRESETS["5V_CMOS"]["v_oh"]
    midpoint = 0.5 * (v_ol + v_oh)

    # Before trigger instant (t = t_step + t_pd = 60 ns), output remains High.
    pre_mask = res.times <= (t_step + t_pd - dt * 0.5)
    assert np.allclose(res.node_voltages["nout"][pre_mask], v_oh, atol=1e-4)

    # Analytic 50% crossing instant: t_cross = t_step + t_pd + t_edge / 2 = 61.0 ns.
    fall_times = _falling_midpoint_times(res.times, res.node_voltages["nout"], midpoint)
    assert len(fall_times) == 1
    assert fall_times[0] == pytest.approx(t_step + t_pd + 0.5 * t_edge, abs=dt)

    # After ramp finishes (t >= 62.5 ns), output settles to V_OL.
    assert res.node_voltages["nout"][-1] == pytest.approx(v_ol, abs=1e-4)

    # Constant conductance matrix: LU built on step 1 (BE) and step 2 (Trapezoidal),
    # then reused across every subsequent digital transition (0 gate-triggered rebuilds).
    assert res.matrix_rebuild_count == 2

    res_pure_trap = MNASolver(circuit).solve_transient(
        duration=80e-9, dt=dt, use_be_on_first_step=False
    )
    assert res_pure_trap.matrix_rebuild_count == 1
    assert not any(w.code == "indeterminate_logic_input" for w in res.warnings)


@pytest.mark.parametrize("n_stages", [3, 5])
def test_ring_oscillator_matches_analytic_period(n_stages: int) -> None:
    """Odd-stage inverter ring oscillates cleanly with period T ~ 2 * N * t_pd."""
    t_pd = 20e-9
    t_edge = 0.5e-9
    dt = 0.25e-9
    expected_period = 2.0 * n_stages * t_pd

    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    nodes = [f"n{i}" for i in range(n_stages)]
    for i in range(n_stages):
        circuit.add(
            LogicGate(
                f"U{i + 1}",
                "not",
                nodes[i],
                nodes[(i + 1) % n_stages],
                family="5V_CMOS",
                t_pd=t_pd,
                t_edge=t_edge,
            )
        )

    duration = 4.5 * expected_period
    res = MNASolver(circuit).solve_transient(duration=duration, dt=dt)
    v_trace = res.node_voltages["n0"]

    assert float(np.max(v_trace)) > 4.5
    assert float(np.min(v_trace)) < 0.5
    # LU rebuilt only for step 1 (BE) -> step 2 (Trapezoidal), never during switching.
    assert res.matrix_rebuild_count == 2

    rises = _rising_midpoint_times(res.times, v_trace, midpoint=2.5)
    assert len(rises) >= 3
    measured_periods = np.diff(rises)
    for period in measured_periods[1:]:
        assert period == pytest.approx(expected_period, rel=0.05)


def test_sr_nand_latch_set_hold_reset_and_recovery() -> None:
    """Cross-coupled active-low NAND SR latch sets, holds state, resets, and recovers."""
    t_pd = 5e-9
    t_edge = 1e-9
    dt = 0.5e-9

    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    vs_s = DCVoltageSource("VS", "s_bar", "0", voltage=0.0)  # Active-low Set
    vs_r = DCVoltageSource("VR", "r_bar", "0", voltage=5.0)
    circuit.add(vs_s)
    circuit.add(vs_r)
    circuit.add(
        LogicGate("U1", "nand", ("s_bar", "q_bar"), "q", family="5V_CMOS", t_pd=t_pd, t_edge=t_edge)
    )
    circuit.add(
        LogicGate("U2", "nand", ("r_bar", "q"), "q_bar", family="5V_CMOS", t_pd=t_pd, t_edge=t_edge)
    )

    solver = MNASolver(circuit)
    solver.reset_transient()

    def advance(steps: int) -> dict[str, float]:
        last_v: dict[str, float] = {}
        for _ in range(steps):
            last_v = solver.step_transient(dt).node_voltages
        return last_v

    # Phase 1: SET (s_bar=0, r_bar=5) => q = High (~4.95 V), q_bar = Low (~0.05 V).
    v = advance(60)
    assert v["q"] == pytest.approx(4.95, abs=0.05)
    assert v["q_bar"] == pytest.approx(0.05, abs=0.05)

    # Phase 2: HOLD (s_bar=5, r_bar=5) => latch remembers q = High, q_bar = Low.
    vs_s.dc_voltage = 5.0
    v = advance(60)
    assert v["q"] == pytest.approx(4.95, abs=0.05)
    assert v["q_bar"] == pytest.approx(0.05, abs=0.05)

    # Phase 3: RESET (s_bar=5, r_bar=0) => q = Low, q_bar = High.
    vs_r.dc_voltage = 0.0
    v = advance(60)
    assert v["q"] == pytest.approx(0.05, abs=0.05)
    assert v["q_bar"] == pytest.approx(4.95, abs=0.05)

    # Phase 4: HOLD after Reset (s_bar=5, r_bar=5) => latch remembers q = Low, q_bar = High.
    vs_r.dc_voltage = 5.0
    v = advance(60)
    assert v["q"] == pytest.approx(0.05, abs=0.05)
    assert v["q_bar"] == pytest.approx(4.95, abs=0.05)


def test_diagnostics_floating_input() -> None:
    """Unwired gate input emits `floating_gate_input` naming gate and pin without crashing."""
    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    circuit.add(DCVoltageSource("VA", "na", "0", voltage=5.0))
    circuit.add(LogicGate("U1", "and", ("na", "nb_open"), "nout"))

    warnings = circuit.validate_topology()
    floating = [w for w in warnings if w.code == "floating_gate_input"]
    assert len(floating) == 1
    assert "U1.b" in floating[0].message
    assert "U1" in floating[0].component_names

    # 10 MOhm input pull-down keeps matrix well-conditioned: 0 V at floating pin => AND = Low.
    res = MNASolver(circuit).solve_dc()
    assert res.node_voltages["nb_open"] == pytest.approx(0.0, abs=1e-6)
    assert res.node_voltages["nout"] == pytest.approx(0.05, abs=1e-5)


def test_diagnostics_indeterminate_input_voltage() -> None:
    """Mid-band steady voltage between V_IL and V_IH emits `indeterminate_logic_input`."""
    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    circuit.add(DCVoltageSource("VDD", "vdd", "0", voltage=5.0))
    circuit.add(Resistor("R1", "vdd", "mid", resistance=10_000.0))
    circuit.add(Resistor("R2", "mid", "0", resistance=10_000.0))
    circuit.add(LogicGate("U1", "not", "mid", "nout", family="5V_CMOS"))

    dc_res = MNASolver(circuit).solve_dc()
    # V(mid) = 2.50 V (negligible 10 MOhm gate loading: 5k // 10M ~ 4.9975k -> 2.4994 V).
    assert 1.5 < dc_res.node_voltages["mid"] < 3.5
    assert any(w.code == "indeterminate_logic_input" for w in dc_res.warnings)

    tr_res = MNASolver(circuit).solve_transient(duration=15e-9, dt=1e-9)
    mid_warnings = [w for w in tr_res.warnings if w.code == "indeterminate_logic_input"]
    assert len(mid_warnings) == 1
    assert "2.50 V" in mid_warnings[0].message


def test_diagnostics_output_contention_analytic_current() -> None:
    """Two opposing totem-pole outputs on one net match analytic contention current."""
    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    circuit.add(DCVoltageSource("VL", "in_low", "0", voltage=0.0))
    circuit.add(DCVoltageSource("VH", "in_high", "0", voltage=5.0))
    u1 = LogicGate("U1", "not", "in_low", "bus", family="5V_CMOS", r_out=100.0)   # Drives V_OH = 4.95 V
    u2 = LogicGate("U2", "not", "in_high", "bus", family="5V_CMOS", r_out=100.0)  # Drives V_OL = 0.05 V
    circuit.add(u1)
    circuit.add(u2)

    res = MNASolver(circuit).solve_dc()
    v_oh = LOGIC_FAMILY_PRESETS["5V_CMOS"]["v_oh"]
    v_ol = LOGIC_FAMILY_PRESETS["5V_CMOS"]["v_ol"]

    # Equal r_out = 100 Ohm => V(bus) = (V_OH + V_OL) / 2 = 2.50 V
    # Analytic contention current I = (V_OH - V_OL) / (R_out1 + R_out2) = 4.90 / 200 = 24.5 mA
    expected_v_bus = 0.5 * (v_oh + v_ol)
    expected_i_contention = (v_oh - v_ol) / (u1.r_out + u2.r_out)

    assert res.node_voltages["bus"] == pytest.approx(expected_v_bus, abs=1e-6)
    assert u1.current == pytest.approx(expected_i_contention, abs=1e-6)
    assert u2.current == pytest.approx(-expected_i_contention, abs=1e-6)

    contention = [w for w in res.warnings if w.code == "output_contention"]
    assert len(contention) == 1
    assert "24.5 mA" in contention[0].message


def test_diagnostics_output_shorted_to_ground_and_supply() -> None:
    """Gate output tied to Ground or ideal DCVoltageSource emits `output_shorted_to_supply`."""
    # Case 1: output shorted to Ground
    c_gnd = Circuit()
    c_gnd.add(Ground("GND1", "0"))
    c_gnd.add(DCVoltageSource("VIN", "nin", "0", voltage=0.0))
    g1 = LogicGate("U1", "not", "nin", "0", family="5V_CMOS", r_out=100.0)
    c_gnd.add(g1)

    res_gnd = MNASolver(c_gnd).solve_dc()
    assert any(w.code == "output_shorted_to_supply" for w in res_gnd.warnings)
    assert g1.current == pytest.approx(4.95 / 100.0, abs=1e-6)

    # Case 2: output shorted to ideal 5 V supply
    c_vcc = Circuit()
    c_vcc.add(Ground("GND1", "0"))
    c_vcc.add(DCVoltageSource("VCC", "vcc", "0", voltage=5.0))
    g2 = LogicGate("U2", "not", "vcc", "vcc", family="5V_CMOS", r_out=100.0)  # Drives V_OL=0.05 against 5V
    c_vcc.add(g2)

    res_vcc = MNASolver(c_vcc).solve_dc()
    short_warnings = [w for w in res_vcc.warnings if w.code == "output_shorted_to_supply"]
    assert len(short_warnings) == 1
    assert "VCC" in short_warnings[0].message
    assert g2.current == pytest.approx((0.05 - 5.0) / 100.0, abs=1e-6)


def test_zero_delay_loop_raises_while_positive_delay_allowed() -> None:
    """Zero-delay feedback cycle raises ZeroDelayLoopError; non-zero t_pd cycle passes."""
    bad = Circuit()
    bad.add(Ground("GND1", "0"))
    bad.add(LogicGate("U1", "not", "n0", "n1", t_pd=0.0))
    bad.add(LogicGate("U2", "not", "n1", "n2", t_pd=0.0))
    bad.add(LogicGate("U3", "not", "n2", "n0", t_pd=0.0))

    with pytest.raises(ZeroDelayLoopError) as exc_info:
        bad.validate_topology()
    assert {"U1", "U2", "U3"}.issubset(set(exc_info.value.gates))

    # Making one gate non-zero delay breaks the zero-delay cycle validly.
    ok = Circuit()
    ok.add(Ground("GND1", "0"))
    ok.add(LogicGate("U1", "not", "n0", "n1", t_pd=10e-9))
    ok.add(LogicGate("U2", "not", "n1", "n2", t_pd=0.0))
    ok.add(LogicGate("U3", "not", "n2", "n0", t_pd=0.0))
    ok.validate_topology()


def test_trapezoidal_rc_load_remains_smooth_without_ringing() -> None:
    """Trapezoidal transient stepping of a gate driving an RC load stays monotonic and bounded."""
    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    circuit.add(
        ACVoltageSource(
            "VPULSE",
            "nin",
            "0",
            waveform="pulse",
            amplitude=5.0,
            frequency=5e6,
            duty_cycle=0.5,
            phase_deg=180.0,
        )
    )
    # Pulse starts at 0V => gate drives High (4.95 V); then at 100 ns jumps to 5V => falls to 0.05 V.
    circuit.add(
        LogicGate("U1", "not", "nin", "nout", family="5V_CMOS", t_pd=10e-9, t_edge=4e-9, r_out=100.0)
    )
    # Capacitor time constant tau = 100 Ohm * 100 pF = 10 ns.
    circuit.add(Capacitor("C1", "nout", "0", capacitance=100e-12, initial_voltage=4.95))

    res = MNASolver(circuit).solve_transient(
        duration=160e-9,
        dt=1e-9,
        method=IntegrationMethod.TRAPEZOIDAL,
    )
    v_out = res.node_voltages["nout"]
    v_ol = LOGIC_FAMILY_PRESETS["5V_CMOS"]["v_ol"]
    v_oh = LOGIC_FAMILY_PRESETS["5V_CMOS"]["v_oh"]

    # Should never ring above V_OH or below V_OL.
    assert float(np.max(v_out)) <= v_oh + 1e-4
    assert float(np.min(v_out)) >= v_ol - 1e-4
    # Discharges smoothly monotonically after t = 110 ns (100 ns step + 10 ns delay).
    post_mask = res.times >= 110e-9
    diffs = np.diff(v_out[post_mask])
    assert np.all(diffs <= 1e-5)
    assert v_out[-1] < 0.2


def test_simulation_session_logic_gate_telemetry_and_diagnostics() -> None:
    """SimulationSession exposes logic_level telemetry, live family presets, and ProblemDiagnostics."""
    circuit = Circuit()
    circuit.add(Ground("GND1", "0"))
    circuit.add(DCVoltageSource("VDD", "vdd", "0", voltage=5.0))
    circuit.add(LogicGate("U1", "and", ("vdd", "open_b"), "nout", family="5V_CMOS"))

    session = SimulationSession(circuit)
    frame = session.start()
    assert frame.state == SimulationState.RUNNING
    assert frame.components["U1"].logic_level == "L"
    assert frame.components["U1"].display_value == "L"
    assert any(d.code == "floating_gate_input" and d.severity == "error" for d in frame.diagnostics)

    # Switch family live and verify thresholds update cleanly.
    session.update_component_parameter("U1", "family", "3V3_LVCMOS")
    u1 = circuit.get_component("U1")
    assert isinstance(u1, LogicGate)
    assert u1.v_oh == pytest.approx(3.2)

    # Zero-delay loop moves session to ERROR state with zero_delay_loop diagnostic.
    bad = Circuit()
    bad.add(Ground("GND1", "0"))
    bad.add(LogicGate("U1", "not", "n0", "n1", t_pd=0.0))
    bad.add(LogicGate("U2", "not", "n1", "n0", t_pd=0.0))
    bad_session = SimulationSession(bad)
    err_frame = bad_session.start()
    assert err_frame.state == SimulationState.ERROR
    assert any(d.code == "zero_delay_loop" and d.severity == "error" for d in err_frame.diagnostics)

    # Ring oscillator stepped incrementally via SimulationSession (start -> step_chunk).
    ring = Circuit()
    ring.add(Ground("GND1", "0"))
    ring.add(LogicGate("U1", "not", "n0", "n1", t_pd=10e-9, t_edge=1e-9))
    ring.add(LogicGate("U2", "not", "n1", "n2", t_pd=10e-9, t_edge=1e-9))
    ring.add(LogicGate("U3", "not", "n2", "n0", t_pd=10e-9, t_edge=1e-9))
    ring_session = SimulationSession(ring, default_dt=1e-9)
    ring_session.start()
    seen_levels = {ring_session.step_chunk(steps=15).components["U1"].logic_level for _ in range(6)}
    assert seen_levels == {"H", "L"}

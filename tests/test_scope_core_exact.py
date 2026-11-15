"""Qt-free exact-value unit tests for `circuitsim.scope` and logic gate capture.

All expected values in this module are derived analytically by hand (with the
formula documented in comments), never copied from code output.

Note on Auto Set (sub-item (c) of MASTER QUEUE A4):
    `ScopeCore.auto_set` (`src/circuitsim/scope/core.py`) currently documents only
    a single-line summary ("Automatically choose vertical scale, offset, timebase,
    and trigger level.") and `PROGRESS.md` does not specify the target peak-to-peak
    division ratio (`Vpp / 5.0`), midpoint rounding, or 1-2-5 log10 snapping tie
    rules. Per task instructions ("read the code's docstring and PROGRESS.md design
    note first. If the rule is not documented, report that and skip this test"),
    an exact-value Auto Set test is intentionally omitted and reported here.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuitsim.components import DCVoltageSource, Ground, LogicGate, Resistor, Switch
from circuitsim.engine import Circuit, SimulationSession
from circuitsim.scope import (
    ChannelConfig,
    SampleBlock,
    ScopeCore,
    ScopeRunState,
    TimebaseConfig,
    TriggerConfig,
    TriggerEdge,
    TriggerMode,
    TriggerStatus,
)
from circuitsim.scope.channels import process_channel_signal
from circuitsim.scope.trigger import find_trigger_crossings


def test_roll_mode_exact_timestamps_and_values() -> None:
    """(a) Roll mode outputs exact trailing window grid ``[t_max - W, t_max)`` and values."""
    # Hand calculation:
    #   seconds_per_div = 0.1 >= 0.1 roll threshold => Roll mode active
    #   W = 10 * 0.1 = 1.0 s, M = 10 * 1 = 10 display points, dt_scope = 0.1 s
    #   Input block: N = 20 samples starting at t0 = 0.0 with dt = 0.1 s:
    #     t_k = 0.1 * k for k = 0..19 => t_min = 0.0 s, t_max = 1.9 s
    #     v_k = 0.5 * k = 5.0 * t_k (volts)
    #   Roll window: t_end = t_max = 1.9 s, t_start = 1.9 - 1.0 = 0.9 s
    #     t_abs[m] = 0.9 + 0.1 * m  = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8] s
    #     t_rel[m] = t_abs[m] - 1.9 = [-1.0, -0.9, ..., -0.1] s
    #     volts[m] = 5.0 * t_abs[m] = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0] V
    tb = TimebaseConfig(seconds_per_div=0.1, points_per_div=1, horizontal_divisions=10)
    assert tb.is_roll_mode
    scope = ScopeCore(timebase=tb)
    scope.assign_probe("CH1", "n1", enabled=True)

    k = np.arange(20, dtype=np.float64)
    v = (0.5 * k).reshape(-1, 1)
    scope.ingest_block(SampleBlock(t0=0.0, dt=0.1, node_names=("n1",), data=v))

    frame = scope.produce_frame()
    assert frame.status == TriggerStatus.ROLL
    assert frame.trigger_time_s is None
    assert frame.run_state == ScopeRunState.RUN

    expected_t_abs = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    expected_t_rel = expected_t_abs - 1.9
    expected_volts = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])

    np.testing.assert_allclose(frame.t_abs, expected_t_abs, atol=1e-12)
    np.testing.assert_allclose(frame.t_rel, expected_t_rel, atol=1e-12)
    assert "CH1" in (frame.traces or {})
    np.testing.assert_allclose(frame.traces["CH1"].volts, expected_volts, atol=1e-12)


def test_single_mode_pre_trigger_capture_freeze_and_ignore_later_samples() -> None:
    """(b) Single mode waits while unarmed/uncrossed, freezes full window on capture, ignores later blocks."""
    # Hand calculation:
    #   Timebase: 1 ms/div, 1 pt/div, 10 div => W = 10 ms, M = 10, dt_scope = 1 ms
    #   Trigger: CH1 rising across level = 2.0 V, pre_trigger_fraction = 0.4
    #     pre = 0.4 * 10 ms = 4.0 ms, post = 0.6 * 10 ms = 6.0 ms
    #   Block 1: t = [0, 1, 2, 3, 4] ms, v = [0.0, 0.5, 1.0, 1.5, 1.8] V (< 2.0 V) -> WAITING
    #   Block 2: t = [5, 6, 7, 8, 9, 10, 11, 12] ms, v = [1.5, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0] V
    #     Rising crossing (5 -> 6 ms, 1.5 -> 2.5 V):
    #       t_trig = 5.0 ms + ((2.0 - 1.5) / (2.5 - 1.5)) * 1.0 ms = 5.5 ms
    #     Post-trigger tail: t_trig + 6.0 ms = 11.5 ms <= t_max = 12.0 ms -> TRIGGERED & STOP
    #     Window [1.5 ms, 11.5 ms) => t_abs = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] ms
    #     Linear interpolation at half-ms points from combined record:
    #       1.5 ms -> 0.5*(0.5 + 1.0) = 0.75 V
    #       2.5 ms -> 0.5*(1.0 + 1.5) = 1.25 V
    #       3.5 ms -> 0.5*(1.5 + 1.8) = 1.65 V
    #       4.5 ms -> 0.5*(1.8 + 1.5) = 1.65 V
    #       5.5 ms -> 0.5*(1.5 + 2.5) = 2.00 V
    #       6.5 ms -> 0.5*(2.5 + 3.0) = 2.75 V
    #       7.5..10.5 ms -> 3.00 V
    tb = TimebaseConfig(seconds_per_div=1e-3, points_per_div=1, horizontal_divisions=10)
    trig = TriggerConfig(
        source="CH1",
        level_v=2.0,
        edge=TriggerEdge.RISING,
        mode=TriggerMode.SINGLE,
        pre_trigger_fraction=0.4,
        hysteresis_div=0.05,
    )
    scope = ScopeCore(timebase=tb, trigger=trig)
    scope.assign_probe("CH1", "n1", enabled=True)
    scope.single()
    assert scope.run_state == ScopeRunState.SINGLE

    b1 = SampleBlock(
        t0=0.0,
        dt=1e-3,
        node_names=("n1",),
        data=np.array([[0.0], [0.5], [1.0], [1.5], [1.8]]),
    )
    scope.ingest_block(b1)
    f1 = scope.produce_frame()
    assert f1.status == TriggerStatus.WAITING
    assert scope.run_state == ScopeRunState.SINGLE

    b2 = SampleBlock(
        t0=5e-3,
        dt=1e-3,
        node_names=("n1",),
        data=np.array([[1.5], [2.5], [3.0], [3.0], [3.0], [3.0], [3.0], [3.0]]),
    )
    scope.ingest_block(b2)
    f2 = scope.produce_frame()
    assert f2.status == TriggerStatus.TRIGGERED
    assert f2.run_state == ScopeRunState.STOP
    assert scope.run_state == ScopeRunState.STOP
    assert f2.trigger_time_s == pytest.approx(5.5e-3, abs=1e-12)

    expected_t_abs = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]) * 1e-3
    expected_t_rel = expected_t_abs - 5.5e-3
    expected_volts = np.array([0.75, 1.25, 1.65, 1.65, 2.00, 2.75, 3.00, 3.00, 3.00, 3.00])

    np.testing.assert_allclose(f2.t_abs, expected_t_abs, atol=1e-12)
    np.testing.assert_allclose(f2.t_rel, expected_t_rel, atol=1e-12)
    np.testing.assert_allclose((f2.traces or {})["CH1"].volts, expected_volts, atol=1e-12)

    # Subsequent block while stopped must be completely ignored and frozen frame preserved.
    b3 = SampleBlock(
        t0=13e-3,
        dt=1e-3,
        node_names=("n1",),
        data=np.full((8, 1), -99.0),
    )
    scope.ingest_block(b3)
    f3 = scope.produce_frame()
    assert scope.run_state == ScopeRunState.STOP
    assert f3 is f2
    np.testing.assert_allclose((f3.traces or {})["CH1"].volts, expected_volts, atol=1e-12)


def test_trigger_hysteresis_requires_dip_below_arm_threshold_to_rearm() -> None:
    """(d) Rising trigger with hysteresis ignores bounce above ``level - hyst_v`` and re-arms below it."""
    # Hand calculation:
    #   level_v = 1.0 V, hysteresis_div = 0.1, volts_per_div = 1.0 V/div => hyst_v = 0.1 V
    #   Arm threshold for rising edge: v <= 1.0 - 0.1 = 0.90 V
    #   t = [0, 1, 2, 3, 4, 5, 6, 7] ms
    #   v = [0.5, 0.5, 1.5, 0.95, 1.5, 0.85, 0.85, 1.15] V
    #   - idx 1->2 (0.5 -> 1.5): armed at idx 0,1 (0.5 <= 0.9), crosses 1.0 at:
    #       t1 = 1 ms + ((1.0 - 0.5) / (1.5 - 0.5)) * 1 ms = 1.5 ms
    #   - idx 3 (0.95 V > 0.90 V): fails to re-arm, so idx 3->4 (0.95 -> 1.5) is ignored!
    #   - idx 5,6 (0.85 V <= 0.90 V): re-arms, so idx 6->7 (0.85 -> 1.15) triggers at:
    #       t2 = 6 ms + ((1.0 - 0.85) / (1.15 - 0.85)) * 1 ms = 6.5 ms
    t = np.arange(8, dtype=np.float64) * 1e-3
    v = np.array([0.5, 0.5, 1.5, 0.95, 1.5, 0.85, 0.85, 1.15], dtype=np.float64)
    cfg = TriggerConfig(
        source="CH1",
        level_v=1.0,
        edge=TriggerEdge.RISING,
        hysteresis_div=0.1,
        holdoff_s=0.0,
    )

    events = find_trigger_crossings(t, v, cfg, volts_per_div=1.0)
    assert len(events) == 2
    assert events[0].time_s == pytest.approx(1.5e-3, abs=1e-12)
    assert events[1].time_s == pytest.approx(6.5e-3, abs=1e-12)


def test_trigger_holdoff_suppresses_crossings_until_holdoff_elapsed() -> None:
    """(e) Holdoff enforces minimum separation between accepted trigger timestamps."""
    # Hand calculation:
    #   Construct symmetric -1 V -> +1 V rising edges centered at t = [1, 2, 3, 4, 5] ms:
    #     t = [0.5, 1.5, 1.6, 2.4, 2.5, 3.5, 3.6, 4.4, 4.5, 5.5] ms (Wait: midpoint of [1.6, 2.4] is 2.0 ms!)
    #     Specifically pairs: (0.5, 1.5)->1.0, (1.6, 2.4)->2.0, (2.5, 3.5)->3.0, (3.6, 4.4)->4.0, (4.5, 5.5)->5.0 ms
    #   holdoff_s = 2.5 ms:
    #     - t = 1.0 ms fires (accepted; last_trig = 1.0 ms)
    #     - t = 2.0 ms: 2.0 - 1.0 = 1.0 ms < 2.5 ms -> suppressed
    #     - t = 3.0 ms: 3.0 - 1.0 = 2.0 ms < 2.5 ms -> suppressed
    #     - t = 4.0 ms: 4.0 - 1.0 = 3.0 ms >= 2.5 ms -> fires (last_trig = 4.0 ms)
    #     - t = 5.0 ms: 5.0 - 4.0 = 1.0 ms < 2.5 ms -> suppressed
    #   Result: exact events at [1.0e-3, 4.0e-3] s (separation 3.0 ms >= 2.5 ms).
    t = np.array([0.5, 1.5, 1.6, 2.4, 2.5, 3.5, 3.6, 4.4, 4.5, 5.5], dtype=np.float64) * 1e-3
    v = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    cfg = TriggerConfig(
        source="CH1",
        level_v=0.0,
        edge=TriggerEdge.RISING,
        hysteresis_div=0.05,
        holdoff_s=2.5e-3,
    )

    events = find_trigger_crossings(t, v, cfg, volts_per_div=1.0)
    assert len(events) == 2
    assert events[0].time_s == pytest.approx(1.0e-3, abs=1e-12)
    assert events[1].time_s == pytest.approx(4.0e-3, abs=1e-12)
    assert events[1].time_s - events[0].time_s == pytest.approx(3.0e-3, abs=1e-12)


def test_10x_probe_with_raw_is_attenuated_tip_false_preserves_node_voltage() -> None:
    """(f) With ``raw_is_attenuated_tip=False``, 10x setting does not double-multiply circuit node voltages."""
    # Hand calculation per `circuitsim/scope/channels.py` docstring:
    #   SampleBlock columns already hold circuit-referred node voltages V_node.
    #   With raw_is_attenuated_tip=False, volts == V_node (NOT 10 * V_node).
    #   With volts_per_div = 2.0 V/div, offset_div = 0.5 div:
    #     raw_v = [2.0, -4.0] V -> volts = [2.0, -4.0] V
    #     display_div_y = volts / 2.0 + 0.5 = [2.0/2.0 + 0.5, -4.0/2.0 + 0.5] = [1.5, -1.5] div
    cfg = ChannelConfig(
        name="CH1",
        enabled=True,
        volts_per_div=2.0,
        offset_div=0.5,
        probe_attenuation=10.0,
    )
    t = np.array([0.0, 1e-3], dtype=np.float64)
    raw = np.array([2.0, -4.0], dtype=np.float64)

    volts, disp_y = process_channel_signal(raw, t, cfg, raw_is_attenuated_tip=False)
    np.testing.assert_allclose(volts, [2.0, -4.0], atol=1e-12)
    np.testing.assert_allclose(disp_y, [1.5, -1.5], atol=1e-12)

    # Also verify through ScopeCore end-to-end (which calls process_channel_signal with default raw_is_attenuated_tip=False).
    tb = TimebaseConfig(seconds_per_div=0.1, points_per_div=1, horizontal_divisions=10)
    scope = ScopeCore(timebase=tb)
    scope.assign_probe("CH1", "n1", enabled=True)
    scope.set_channel("CH1", volts_per_div=2.0, offset_div=0.5, probe_attenuation=10.0)
    t_blk = np.full((12, 1), 2.0, dtype=np.float64)
    t_blk[:6, 0] = -4.0
    scope.ingest_block(SampleBlock(t0=0.0, dt=0.1, node_names=("n1",), data=t_blk))
    frame = scope.produce_frame()
    tr = (frame.traces or {})["CH1"]
    assert tr.volts[0] == pytest.approx(-4.0, abs=1e-12)
    assert tr.volts[-1] == pytest.approx(2.0, abs=1e-12)
    assert tr.display_div_y[0] == pytest.approx(-1.5, abs=1e-12)
    assert tr.display_div_y[-1] == pytest.approx(1.5, abs=1e-12)


def test_logic_not_gate_in_circuit_scope_high_low_match_v_oh_and_v_ol() -> None:
    """(g) NOT gate circuit in SimulationSession + ScopeCore gives exact v_oh high and v_ol low."""
    # Hand calculation (5V_CMOS family in `circuitsim.components.logic_gate`):
    #   v_il = 1.5 V, v_ih = 3.5 V, v_ol = 0.05 V, v_oh = 4.95 V
    #   r_out = 100.0 ohm, GMIN = 1e-12 S -> loading factor 1 / (1 + 100 * 1e-12) = 1 - 1e-10
    #   Before S1 closes: gin tied to ground via R1 (1 kOhm) + r_in (10 MOhm) -> V(gin) = 0.0 V < v_il
    #     => steady V(gout) = gate.v_oh = 4.95 V (within 1e-9 V)
    #   After S1 closes: gin connected to 5.0 V source -> V(gin) = 5.0 V > v_ih
    #     => after t_pd (10 ns) + effective_edge (~20 ns at dt=10 ns), V(gout) settles to gate.v_ol = 0.05 V
    c = Circuit()
    c.add(DCVoltageSource("V1", "vcc", "0", voltage=5.0))
    c.add(Switch("S1", "vcc", "gin", closed=False))
    c.add(Resistor("R1", "gin", "0", resistance=1000.0))
    gate = LogicGate("U1", "not", "gin", "gout", family="5V_CMOS")
    c.add(gate)
    c.add(Ground("G1", "0"))

    session = SimulationSession(c, default_dt=10e-9)
    scope = ScopeCore(
        timebase=TimebaseConfig(seconds_per_div=500e-9, points_per_div=100, horizontal_divisions=10),
        trigger=TriggerConfig(
            source="CH1",
            level_v=2.5,
            edge=TriggerEdge.FALLING,
            mode=TriggerMode.NORMAL,
            pre_trigger_fraction=0.5,
        ),
    )
    scope.assign_probe("CH1", "gout", enabled=True)
    scope.bind_session(session)

    session.start()
    session.step_chunk(steps=300, dt=10e-9)  # 3.0 us high at gate.v_oh
    session.toggle_switch("S1")
    session.step_chunk(steps=300, dt=10e-9)  # 3.0 us -> crosses 2.5 V falling, settles to gate.v_ol

    frame = scope.produce_frame()
    assert frame.status == TriggerStatus.TRIGGERED
    ch1 = (frame.traces or {})["CH1"]
    assert ch1.measurements.vmax == pytest.approx(gate.v_oh, abs=1e-6)
    assert ch1.measurements.vmin == pytest.approx(gate.v_ol, abs=1e-6)

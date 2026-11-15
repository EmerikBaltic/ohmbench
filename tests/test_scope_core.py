"""Comprehensive Qt-free unit and circuit-integration tests for `circuitsim.scope`."""

from __future__ import annotations

import math
import pickle
import subprocess
import sys

import numpy as np
import pytest

from conftest import assert_module_has_no_qt_imports
from circuitsim.components import (
    ACVoltageSource,
    Capacitor,
    DCVoltageSource,
    Ground,
    LED,
    LogicGate,
    Resistor,
    Switch,
)
from circuitsim.engine import (
    Circuit,
    MNASolver,
    SimulationSession,
    SimulationSpeed,
)
from circuitsim.scope import (
    CHANNEL_COLORS,
    ChannelConfig,
    ChannelCoupling,
    MathOp,
    RingBuffer,
    SampleBlock,
    SampleRecorder,
    ScopeCore,
    ScopeRunState,
    TimebaseConfig,
    TriggerConfig,
    TriggerCoupling,
    TriggerEdge,
    TriggerMode,
    TriggerStatus,
    compute_coupled_solver_dt,
    compute_cursors,
    compute_fft,
    compute_math_trace,
    compute_xy,
    find_trigger_crossings,
    measure_channel,
    measure_phase_lag_deg,
    process_channel_signal,
    resample_for_display,
)


def _build_rc_ac_circuit(*, freq_hz: float = 1000.0, amplitude: float = 5.0, c_farads: float = 1e-6) -> Circuit:
    c = Circuit()
    c.add(ACVoltageSource("V1", "in", "0", amplitude=amplitude, frequency=freq_hz))
    c.add(Resistor("R1", "in", "out", resistance=1000.0))
    c.add(Capacitor("C1", "out", "0", capacitance=c_farads))
    c.add(Ground("G1", "0"))
    return c


def test_scope_package_has_no_qt_imports() -> None:
    """All `circuitsim.scope` modules must load without Qt bindings or `scope_demo`."""
    assert_module_has_no_qt_imports(
        "circuitsim.scope",
        "circuitsim.scope.timebase",
        "circuitsim.scope.capture",
        "circuitsim.scope.trigger",
        "circuitsim.scope.channels",
        "circuitsim.scope.measurements",
        "circuitsim.scope.math_fft",
        "circuitsim.scope.core",
    )
    # Verify in a clean subprocess that importing `circuitsim.scope` never pulls in `tools.scope_demo`
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, circuitsim.scope, circuitsim.scope.core; "
                "assert 'circuitsim.tools.scope_demo' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert res.returncode == 0, res.stderr


def test_sample_block_and_scope_frame_immutability_and_pickling() -> None:
    """All output NumPy arrays remain read-only (`write=False`) before and after pickling."""
    t = np.linspace(0.0, 4e-3, 400, endpoint=False)
    v = 3.0 * np.sin(2.0 * math.pi * 1000.0 * t)
    block = SampleBlock(
        t0=0.0,
        dt=float(t[1] - t[0]),
        node_names=("in", "0"),
        data=np.column_stack((v, np.zeros_like(v))),
    )
    assert not block.data.flags.writeable
    with pytest.raises(ValueError):
        block.data[0, 0] = 99.0

    unp_block = pickle.loads(pickle.dumps(block))
    assert not unp_block.data.flags.writeable
    assert np.allclose(unp_block.data, block.data)

    scope = ScopeCore()
    scope.assign_probe("CH1", "in", enabled=True)
    scope.assign_probe("CH2", "0", enabled=True)
    scope.set_timebase(400e-6)
    scope.set_math(True, MathOp.SUB)
    scope.set_fft("CH1")
    scope.set_xy(True)
    scope.ingest_block(block)
    frame = scope.produce_frame()

    assert not frame.t_rel.flags.writeable
    assert not frame.t_abs.flags.writeable
    assert frame.math_trace is not None and not frame.math_trace.flags.writeable
    assert frame.fft is not None and not frame.fft.freqs_hz.flags.writeable
    assert frame.xy is not None and not frame.xy.x.flags.writeable

    unp_frame = pickle.loads(pickle.dumps(frame))
    assert not unp_frame.t_rel.flags.writeable
    assert unp_frame.math_trace is not None and not unp_frame.math_trace.flags.writeable
    assert unp_frame.fft is not None and not unp_frame.fft.magnitude_v.flags.writeable
    assert unp_frame.xy is not None and not unp_frame.xy.y.flags.writeable
    assert not unp_frame.traces["CH1"].volts.flags.writeable
    assert not unp_frame.traces["CH1"].volts_max.flags.writeable


def test_9a_solver_to_scope_sample_parity() -> None:
    """`SimulationSession` sink matches raw `MNASolver.solve_transient()` node voltages within `1e-12`."""
    dt = 10e-6
    steps = 50
    raw_res = MNASolver(_build_rc_ac_circuit()).solve_transient(steps * dt, dt=dt)

    session = SimulationSession(_build_rc_ac_circuit(), default_dt=dt)
    rec = SampleRecorder()
    session.attach_scope_sink(rec)
    session.step_chunk(steps - 1)  # start() advances step 1, step_chunk advances remaining 49

    t_arr, data_arr, names = rec.concatenated()
    assert len(t_arr) == steps
    assert np.max(np.abs(t_arr - raw_res.times[1:])) <= 1e-15
    assert np.max(np.abs(data_arr[:, names.index("in")] - raw_res.node_voltages["in"][1:])) <= 1e-12
    assert np.max(np.abs(data_arr[:, names.index("out")] - raw_res.node_voltages["out"][1:])) <= 1e-12


def test_9b_decoupled_time_speeds_and_overflow_with_injected_clock() -> None:
    """Decoupled wall clock supports 1x/0.1x/10x/max speeds and flags overflow without blocking."""
    now = [0.0]

    def fake_clock() -> float:
        now[0] += 1e-6  # 1 us per clock check (10000 checks = 10 ms)
        return now[0]

    session = SimulationSession(_build_rc_ac_circuit(), default_dt=1e-5, clock=fake_clock)
    session.start()

    # 10x speed for 1 ms wall = 10 ms sim -> 1000 steps at dt=10 us; generous budget 1.0 s
    session.speed = SimulationSpeed.FAST
    f_fast = session.step_for_wall_elapsed(0.001, max_wall_budget_s=1.0)
    assert not f_fast.overflow
    t_after_fast = f_fast.sim_time

    # 0.1x speed for 1 ms wall = 0.1 ms sim -> 10 steps at dt=10 us
    session.speed = SimulationSpeed.SLOW_MOTION
    f_slow = session.step_for_wall_elapsed(0.001, max_wall_budget_s=1.0)
    assert not f_slow.overflow
    # Hand ratio: 1000 steps / 10 steps = 100x difference between FAST (10x) and SLOW (0.1x)
    adv_fast = t_after_fast - 1e-5
    adv_slow = f_slow.sim_time - t_after_fast
    assert abs(adv_fast / adv_slow - 100.0) < 1.0

    # Budget overrun under 1x sets overflow=True instead of hanging
    session.speed = SimulationSpeed.REAL_TIME
    f_over = session.step_for_wall_elapsed(10.0, max_wall_budget_s=1e-4)
    assert f_over.overflow

    # AS_FAST_AS_POSSIBLE executes continuously up to wall budget and never raises overflow
    session.speed = SimulationSpeed.AS_FAST_AS_POSSIBLE
    f_max = session.step_for_wall_elapsed(1.0, max_wall_budget_s=1e-4)
    assert not f_max.overflow


def test_9c_step_coupling_and_peak_detect_resampling() -> None:
    """`dt = min(dt_scope, dt_circuit)`, single LU rebuild on timebase change, peak-detect & ring wrap."""
    session = SimulationSession(_build_rc_ac_circuit(freq_hz=5000.0, c_farads=1e-6))
    session.start()
    dt_circ = session.solver.choose_default_dt(0.02)

    # At 10 ms/div -> dt_scope = 100 us > dt_circuit -> coupled dt = dt_circuit
    dt_slow_tb = session.set_timebase(10e-3)
    assert abs(dt_slow_tb - compute_coupled_solver_dt(10e-3, dt_circ)) < 1e-15
    assert dt_slow_tb <= dt_circ

    # Tighter 100 us/div -> dt_scope = 1 us < dt_circuit -> rebuilds LU once then caches
    dt_fast_tb = session.set_timebase(100e-6)
    assert abs(dt_fast_tb - 1e-6) < 1e-15
    sim_before = session.solver.sim_time
    r1 = session.solver.step_transient(session.dt)
    r2 = session.solver.step_transient(session.dt)
    assert r1.rebuilt_matrix and not r2.rebuilt_matrix
    assert abs(r1.sim_time - (sim_before + 1e-6)) < 1e-15

    # Single-sample glitch placed midway between display grid samples is caught in peak-detect envelope
    t = np.linspace(0.0, 1e-3, 10001)
    y = np.sin(2.0 * math.pi * 1000.0 * t)
    y[505] = 9.25  # between display points at index 50 and 51
    tr = resample_for_display(t, y, 0.0, 1e-3, num_points=1000)
    assert tr.peak_detect
    assert np.max(tr.y_max) >= 9.25
    assert np.all(tr.y_min <= tr.y) and np.all(tr.y <= tr.y_max)

    # Circular RingBuffer wrap-around keeps latest chronological tail cleanly
    rb = RingBuffer(capacity=5)
    rb.append_block(SampleBlock(t0=0.0, dt=1.0, node_names=("a",), data=np.arange(4.0).reshape(-1, 1)))
    rb.append_block(SampleBlock(t0=4.0, dt=1.0, node_names=("a",), data=np.arange(4.0, 8.0).reshape(-1, 1)))
    rt, rd, _ = rb.get_arrays()
    assert np.allclose(rt, np.array([3.0, 4.0, 5.0, 6.0, 7.0]))
    assert np.allclose(rd[:, 0], np.array([3.0, 4.0, 5.0, 6.0, 7.0]))


def test_9d_trigger_rising_falling_either_hysteresis_holdoff_ac_auto_normal() -> None:
    """Trigger detects all edges with <0.1*dt precision, hysteresis noise rejection, holdoff, AC & Auto/Normal."""
    dt = 10e-6
    t = np.arange(0, 5e-3, dt)
    t_shift = 13.7e-6
    v_clean = 2.0 * np.sin(2.0 * math.pi * 1000.0 * (t - t_shift))

    # Sub-sample rising crossing at 1 ms + 13.7 us (hand formula: sin(0) crossing = k*T + t_shift)
    ev_r = find_trigger_crossings(
        t, v_clean, TriggerConfig(edge=TriggerEdge.RISING, hysteresis_div=0.05)
    )
    assert abs(ev_r[1].time_s - (1e-3 + t_shift)) < 0.1 * dt

    # Falling crossing at 0.5 ms + 13.7 us
    ev_f = find_trigger_crossings(
        t, v_clean, TriggerConfig(edge=TriggerEdge.FALLING, hysteresis_div=0.05)
    )
    assert abs(ev_f[0].time_s - (0.5e-3 + t_shift)) < 0.1 * dt

    # Either captures both slopes cleanly starting with rising at t=13.7 us
    ev_e = find_trigger_crossings(
        t, v_clean, TriggerConfig(edge=TriggerEdge.EITHER, hysteresis_div=0.05)
    )
    assert [e.edge for e in ev_e[:3]] == [TriggerEdge.RISING, TriggerEdge.FALLING, TriggerEdge.RISING]

    # Hysteresis 0.05 div * 1 V/div = 50 mV rejects +-25 mV zero-crossing chatter
    rng = np.random.default_rng(7)
    v_noisy = 2.0 * np.sin(2.0 * math.pi * 1000.0 * t + 0.25) + rng.uniform(-0.025, 0.025, len(t))
    ev_hyst = find_trigger_crossings(
        t, v_noisy, TriggerConfig(edge=TriggerEdge.RISING, hysteresis_div=0.05), volts_per_div=1.0
    )
    assert len(ev_hyst) == 5

    # Holdoff 1.5 ms skips every odd 1 ms cycle
    ev_hold = find_trigger_crossings(
        t, v_clean, TriggerConfig(edge=TriggerEdge.RISING, holdoff_s=1.5e-3)
    )
    assert len(ev_hold) == 3

    # AC trigger coupling on 5 V DC + 1 V sine triggers at 0 V despite non-integer 3.4 cycles
    v_dc_bias = 5.0 + 1.0 * np.sin(2.0 * math.pi * 1000.0 * t[:340])
    ev_ac = find_trigger_crossings(
        t[:340], v_dc_bias, TriggerConfig(level_v=0.0, coupling=TriggerCoupling.AC)
    )
    assert len(ev_ac) == 3

    # Normal stays WAITING on flat DC; Auto enters AUTO_FREE_RUN after simulated timeout
    dc_block_short = SampleBlock(t0=0.0, dt=1e-3, node_names=("n1",), data=np.full((20, 1), 3.3))
    dc_block_long = SampleBlock(t0=0.02, dt=1e-3, node_names=("n1",), data=np.full((150, 1), 3.3))

    scope_norm = ScopeCore(
        timebase=TimebaseConfig(seconds_per_div=1e-3),
        trigger=TriggerConfig(mode=TriggerMode.NORMAL, level_v=1.5),
    )
    scope_norm.assign_probe("CH1", "n1", enabled=True)
    scope_norm.ingest_block(dc_block_short)
    scope_norm.ingest_block(dc_block_long)
    assert scope_norm.produce_frame().status == TriggerStatus.WAITING

    scope_auto = ScopeCore(
        timebase=TimebaseConfig(seconds_per_div=1e-3),
        trigger=TriggerConfig(mode=TriggerMode.AUTO, level_v=1.5, auto_timeout_s=0.1),
    )
    scope_auto.assign_probe("CH1", "n1", enabled=True)
    scope_auto.ingest_block(dc_block_short)
    assert scope_auto.produce_frame().status == TriggerStatus.WAITING
    scope_auto.ingest_block(dc_block_long)
    f_auto = scope_auto.produce_frame()
    assert f_auto.status == TriggerStatus.AUTO_FREE_RUN
    assert abs(f_auto.traces["CH1"].measurements.vavg - 3.3) < 1e-9


def test_9e_rc_step_single_shot_freeze_and_rise_time() -> None:
    """Single-shot trigger captures an RC step, checks `t_r = ln(9)*RC`, and freezes at STOP."""
    # Analytical RC step: tau = R * C = 1000 * 1e-6 = 1 ms; t_r(10%-90%) = ln(9) * tau ≈ 2.1972 ms
    tau_s = 1000.0 * 1e-6
    expected_rise = math.log(9.0) * tau_s

    c = Circuit()
    c.add(DCVoltageSource("V1", "src", "0", voltage=5.0))
    c.add(Switch("SW1", "src", "in", closed=False))
    c.add(Resistor("R1", "in", "out", resistance=1000.0))
    c.add(Capacitor("C1", "out", "0", capacitance=1e-6))
    c.add(Resistor("R_bleed", "out", "0", resistance=1e8))
    c.add(Ground("G1", "0"))

    session = SimulationSession(c, default_dt=10e-6)
    scope = ScopeCore()
    scope.assign_probe("CH1", "out", enabled=True)
    scope.set_timebase(1e-3)
    scope.set_trigger(source="CH1", level_v=2.0, edge=TriggerEdge.RISING, pre_trigger_fraction=0.2)
    scope.bind_session(session)

    # Settle initial 0 V with switch open, then arm SINGLE before closing switch
    session.step_chunk(200)
    scope.clear()
    scope.single()
    session.toggle_switch("SW1")
    session.step_chunk(1000)  # 10 ms > 8 ms post-trigger tail

    f1 = scope.produce_frame()
    assert scope.run_state == ScopeRunState.STOP
    assert f1.status == TriggerStatus.TRIGGERED
    t_rise_meas = f1.traces["CH1"].measurements.rise_time_s
    assert t_rise_meas is not None
    assert abs(t_rise_meas - expected_rise) / expected_rise < 0.03

    # Subsequent chunks MUST NOT mutate the frozen single-shot frame
    session.step_chunk(500)
    f2 = scope.produce_frame()
    assert f2.trigger_time_s == f1.trigger_time_s
    assert np.array_equal(f2.traces["CH1"].volts, f1.traces["CH1"].volts)


def test_9f_rc_lowpass_phase_lag_and_attenuation_end_to_end() -> None:
    """At cutoff frequency `f = f_c = 1/(2*pi*R*C)`, RC filter has |H|=1/sqrt(2) and +45 deg phase lag."""
    f_c = 1000.0
    r_ohm = 1000.0
    c_farads = 1.0 / (2.0 * math.pi * f_c * r_ohm)  # 159.15494 nF
    amp = 5.0

    session = SimulationSession(_build_rc_ac_circuit(freq_hz=f_c, amplitude=amp, c_farads=c_farads))
    scope = ScopeCore()
    scope.assign_probe("CH1", "in", enabled=True)
    scope.assign_probe("CH2", "out", enabled=True)
    scope.set_channel("CH1", volts_per_div=2.0)
    scope.set_channel("CH2", volts_per_div=2.0)
    scope.bind_session(session)
    scope.set_timebase(1e-3)
    session.step_chunk(6000)  # advance cleanly into steady state

    frame = scope.produce_frame()
    m_in = frame.traces["CH1"].measurements
    m_out = frame.traces["CH2"].measurements
    assert m_in.frequency_hz is not None and abs(m_in.frequency_hz - f_c) < 1.0
    # Hand formula: Vpp_out = 2*amp / sqrt(2) = 7.07107 V; Vrms_out = amp / 2 = 2.500 V
    assert abs(m_out.vpp - (2.0 * amp / math.sqrt(2.0))) < 0.05
    assert abs(m_out.vrms - (amp / 2.0)) < 0.03
    assert frame.phase_lag_ch1_ch2_deg is not None
    assert abs(frame.phase_lag_ch1_ch2_deg - 45.0) < 0.5


def test_9g_differential_math_ch1_minus_ch2_resistor_current() -> None:
    """Math trace `CH1 - CH2` recovers exact floating resistor drop across `R1` for current calc."""
    c = Circuit()
    c.add(ACVoltageSource("V1", "n_in", "0", amplitude=10.0, frequency=1000.0))
    c.add(Resistor("R1", "n_in", "n_mid", resistance=100.0))
    c.add(Resistor("R2", "n_mid", "0", resistance=100.0))
    c.add(Ground("G1", "0"))

    session = SimulationSession(c)
    scope = ScopeCore()
    scope.assign_probe("CH1", "n_in", enabled=True)
    scope.assign_probe("CH2", "n_mid", enabled=True)
    scope.set_math(True, MathOp.SUB, source_a="CH1", source_b="CH2")
    scope.bind_session(session)
    scope.set_timebase(500e-6)
    session.step_chunk(1200)

    frame = scope.produce_frame()
    assert frame.math_trace is not None
    # Hand formula: equal divider R1=R2=100 => V(R1) = 0.5 * V_in => 5 V peak => 10 Vpp, I_peak = 50 mA
    m_diff = measure_channel(frame.t_abs, frame.math_trace)
    assert abs(m_diff.vpp - 10.0) < 0.05
    assert abs((0.5 * m_diff.vpp) / 100.0 - 0.05) < 0.0005

    sum_t = compute_math_trace(frame.traces["CH1"].volts, frame.traces["CH2"].volts, MathOp.ADD)
    mul_t = compute_math_trace(frame.traces["CH1"].volts, frame.traces["CH2"].volts, MathOp.MUL)
    assert np.allclose(sum_t, frame.traces["CH1"].volts + frame.traces["CH2"].volts)
    assert np.allclose(mul_t, frame.traces["CH1"].volts * frame.traces["CH2"].volts)


def test_9h_channels_10x_probe_gnd_invert_and_ac_integer_period() -> None:
    """Validate channel bounds, 4 fixed hex colors, 1x/10x probe rule, GND, invert & AC integer-period mean."""
    assert len(CHANNEL_COLORS) == 4
    with pytest.raises(ValueError):
        ChannelConfig(name="CH1", volts_per_div=5e-4)  # < 1 mV/div
    with pytest.raises(ValueError):
        ChannelConfig(name="CH1", volts_per_div=2000.0)  # > 1 kV/div
    with pytest.raises(ValueError):
        ChannelConfig(name="CH1", probe_attenuation=5.0)

    t = np.linspace(0.0, 3.4e-3, 3400)
    raw = 5.0 + 2.0 * np.sin(2.0 * math.pi * 1000.0 * t)
    ch1x = ChannelConfig(name="CH1", enabled=True, volts_per_div=1.0, offset_div=1.0, coupling=ChannelCoupling.AC, probe_attenuation=1.0)
    ch10x = ChannelConfig(name="CH1", enabled=True, volts_per_div=1.0, offset_div=1.0, coupling=ChannelCoupling.AC, probe_attenuation=10.0)
    v1x, d1x = process_channel_signal(raw, t, ch1x)
    v10x, d10x = process_channel_signal(raw / 10.0, t, ch10x, raw_is_attenuated_tip=True)
    assert np.allclose(v1x, v10x, atol=1e-12)
    assert np.allclose(d1x, d10x, atol=1e-12)
    # 3 integer periods inside [0, 3 ms] -> zero mean, exact +-2 V AC amplitude
    assert abs(float(np.mean(v1x[:3000]))) < 1e-4
    # Fallback on < 1 period removes full window mean
    v_short, _ = process_channel_signal(raw[:400], t[:400], ch1x)
    assert abs(float(np.mean(v_short))) < 1e-12

    v_gnd, d_gnd = process_channel_signal(raw, t, ChannelConfig(name="CH2", enabled=True, offset_div=-1.5, coupling=ChannelCoupling.GND))
    assert np.all(v_gnd == 0.0) and np.allclose(d_gnd, -1.5)
    v_inv, _ = process_channel_signal(raw, t, ChannelConfig(name="CH3", enabled=True, invert=True))
    assert np.allclose(v_inv, -raw)


def test_9i_fft_hann_window_coherent_and_offbin_accuracy_and_harmonics() -> None:
    """Periodic Hann FFT hits <0.5 dB on coherent and off-bin tones and square-wave odd harmonics."""
    fs = 10_000.0
    n = 1024
    t = np.arange(n) / fs
    df = fs / n

    # 1) Half-bin off-grid sine (worst-case scalloping bin offset)
    f_off = 100.5 * df
    amp = 2.0
    res_off = compute_fft(t, amp * np.sin(2.0 * math.pi * f_off * t), window="hann")
    assert res_off.fundamental_dbv is not None and res_off.fundamental_freq_hz is not None
    assert abs(res_off.fundamental_dbv - 20.0 * math.log10(amp)) < 0.5
    assert abs(res_off.fundamental_freq_hz - f_off) < df

    # 2) Square wave Fourier series: A1 = 4*A/pi, A3 = 4*A/(3*pi) = -9.542 dB relative to A1
    f_sq = 32 * df  # coherent integer cycles
    v_sq = np.sign(np.sin(2.0 * math.pi * f_sq * t))
    res_sq = compute_fft(t, v_sq, window="hann")
    idx_1 = 32
    idx_3 = 96
    assert abs(res_sq.magnitude_v[idx_1] - (4.0 / math.pi)) < 0.05
    ratio_db = res_sq.magnitude_dbv[idx_3] - res_sq.magnitude_dbv[idx_1]
    assert abs(ratio_db - (-20.0 * math.log10(3.0))) < 0.3


def test_9j_roll_mode_auto_set_xy_and_cursors() -> None:
    """Verify Roll mode >=100 ms/div, Auto Set framing, XY circle radius, and cursor delta readout."""
    t = np.linspace(0.0, 0.25, 2500, endpoint=False)
    v = 5.0 * np.sin(2.0 * math.pi * 1000.0 * t)
    scope = ScopeCore()
    scope.assign_probe("CH1", "n1", enabled=True)
    scope.set_channel("CH1", volts_per_div=50.0, offset_div=3.0)
    scope.set_timebase(100e-3)  # Roll mode (>= 100 ms/div)
    scope.ingest_block(SampleBlock(t0=0.0, dt=float(t[1] - t[0]), node_names=("n1",), data=v.reshape(-1, 1)))
    f_roll = scope.produce_frame()
    assert f_roll.status == TriggerStatus.ROLL
    assert abs(f_roll.t_abs[-1] - f_roll.sim_time) <= 2.0 * scope.timebase.dt_scope

    # Auto Set chooses sensible ~2 V/div for 10 Vpp, centered offset, ~2.5 cycles (~250 us -> 200 us/div)
    scope.set_timebase(50e-3)
    scope.auto_set()
    assert scope.channels["CH1"].volts_per_div in (1.0, 2.0)
    assert abs(scope.channels["CH1"].offset_div) <= 0.5
    assert scope.timebase.seconds_per_div in (200e-6, 500e-6)

    # XY Lissajous circle on quadrature sines
    t_xy = np.linspace(0.0, 1e-3, 500, endpoint=False)
    vx = 3.0 * np.cos(2.0 * math.pi * 1000.0 * t_xy)
    vy = 3.0 * np.sin(2.0 * math.pi * 1000.0 * t_xy)
    xy = compute_xy(vx, vy)
    assert np.max(np.abs((xy.x ** 2 + xy.y ** 2) - 9.0)) < 1e-12

    cur = compute_cursors(t_xy, vy, 0.0, 0.25e-3)
    assert abs(cur.dt - 0.25e-3) < 1e-12
    assert abs(cur.dv - 3.0) < 0.02
    assert abs(cur.inv_dt_hz - 4000.0) < 1e-6


def test_9k_measurements_edge_cases_and_reasons() -> None:
    """Flat DC, empty input, and partial-cycle signals report clear explicit reason strings."""
    m_empty = measure_channel(np.empty(0), np.empty(0))
    assert m_empty.reason == "empty signal" and m_empty.frequency_hz is None

    t = np.linspace(0.0, 1e-3, 100)
    m_dc = measure_channel(t, np.full_like(t, -2.5))
    assert m_dc.vpp == 0.0 and m_dc.vavg == -2.5 and m_dc.vrms == 2.5
    assert m_dc.frequency_hz is None and m_dc.reason == "no complete period"

    phase, p_reason = measure_phase_lag_deg(t, np.full_like(t, 1.0), np.full_like(t, 2.0))
    assert phase is None and p_reason == "insufficient amplitude"


def test_9l_led_and_logic_gate_circuit_waveforms() -> None:
    """Real project nonlinear LED circuit and digital NOT gate waveforms work cleanly end-to-end."""
    # 1) LED half-wave clamping: 5 V AC -> 150 Ohm -> Red LED (Vf=2.0 V) -> 0
    # Forward peak anode voltage stays clamped near +2 V while reverse cycle swings to -5 V (no current)
    c_led = Circuit()
    c_led.add(ACVoltageSource("V1", "src", "0", amplitude=5.0, frequency=1000.0))
    c_led.add(Resistor("R1", "src", "anode", resistance=150.0))
    c_led.add(LED("D1", "anode", "0", color_name="Red", forward_voltage=2.0))
    c_led.add(Ground("G1", "0"))

    s_led = SimulationSession(c_led, default_dt=10e-6)
    sc_led = ScopeCore()
    sc_led.assign_probe("CH1", "src", enabled=True)
    sc_led.assign_probe("CH2", "anode", enabled=True)
    sc_led.bind_session(s_led)
    sc_led.set_timebase(500e-6)
    s_led.step_chunk(1200)
    f_led = sc_led.produce_frame()
    m_anode = f_led.traces["CH2"].measurements
    assert m_anode.vmax < 2.5  # forward LED clamp well below +5 V supply peak
    assert m_anode.vmin < -4.5  # reverse swing approaches -5 V

    # 2) LogicGate NOT transition via closed/open Switch step
    c_gate = Circuit()
    c_gate.add(DCVoltageSource("VDD", "vdd", "0", voltage=5.0))
    c_gate.add(Switch("SW1", "vdd", "gin", closed=False))
    c_gate.add(Resistor("R_pd", "gin", "0", resistance=1000.0))
    c_gate.add(LogicGate("U1", "not", "gin", "gout", family="5V_CMOS"))
    c_gate.add(Ground("G1", "0"))

    s_gate = SimulationSession(c_gate)
    sc_gate = ScopeCore()
    sc_gate.assign_probe("CH1", "gin", enabled=True)
    sc_gate.assign_probe("CH2", "gout", enabled=True)
    sc_gate.set_trigger(source="CH1", level_v=2.0, edge=TriggerEdge.RISING, pre_trigger_fraction=0.2)
    sc_gate.bind_session(s_gate)
    sc_gate.set_timebase(100e-9)  # 100 ns/div (W = 1 us) resolves ns gate propagation delay cleanly
    s_gate.step_chunk(300)  # capture pre-toggle low baseline (gin = 0 V, gout = 4.95 V)
    sc_gate.single()
    s_gate.toggle_switch("SW1")  # gin steps 0 V -> 5 V; gout falls ~4.95 V -> ~0.05 V after t_pd
    s_gate.step_chunk(1500)
    f_gate = sc_gate.produce_frame()
    assert f_gate.status == TriggerStatus.TRIGGERED
    m_gout = f_gate.traces["CH2"].measurements
    assert m_gout.vmax > 4.5 and m_gout.vmin < 0.2

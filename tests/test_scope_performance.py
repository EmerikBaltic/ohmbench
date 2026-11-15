"""Performance benchmarks for `circuitsim.scope` and `SimulationSession` sink overhead."""

from __future__ import annotations

import gc
import math
import statistics
import time

from PySide6.QtCore import QEventLoop, QPointF, QTimer

from circuitsim.components import (
    ACVoltageSource,
    Capacitor,
    DCVoltageSource,
    Ground,
    Inductor,
    LED,
    Resistor,
    Switch,
)
from circuitsim.engine import Circuit, SimulationSession
from circuitsim.scope import MathOp, ScopeCore


def _build_circuit_a_rc() -> Circuit:
    c = Circuit()
    c.add(ACVoltageSource("V1", "in", "0", amplitude=5.0, frequency=1000.0))
    c.add(Resistor("R1", "in", "n1", resistance=1000.0))
    c.add(Resistor("R2", "n1", "out", resistance=10000.0))
    c.add(Capacitor("C1", "out", "0", capacitance=1e-6))
    c.add(Ground("G1", "0"))
    return c


def _build_circuit_b_ladder() -> Circuit:
    c = Circuit()
    c.add(ACVoltageSource("V1", "n0", "0", amplitude=5.0, frequency=1000.0))
    c.add(Ground("G1", "0"))
    for k in range(15):
        a = f"n{k}"
        b = f"n{k + 1}"
        c.add(Resistor(f"Rs{k}", a, b, resistance=100.0))
        c.add(Resistor(f"Rp{k}", b, "0", resistance=1000.0))
    c.add(Capacitor("C_load", "n15", "0", capacitance=1e-6))
    return c


def _build_circuit_c_rl_switch() -> Circuit:
    c = Circuit()
    c.add(DCVoltageSource("V1", "src", "0", voltage=12.0))
    c.add(Switch("SW1", "src", "n1", closed=True))
    c.add(Resistor("R1", "n1", "out", resistance=100.0))
    c.add(Inductor("L1", "out", "0", inductance=10e-3, dcr=1.0))
    c.add(Ground("G1", "0"))
    return c


def _build_circuit_d_led() -> Circuit:
    c = Circuit()
    c.add(ACVoltageSource("V1", "src", "0", amplitude=5.0, frequency=1000.0))
    c.add(Resistor("R1", "src", "anode", resistance=150.0))
    c.add(LED("D1", "anode", "0", color_name="Red"))
    c.add(Ground("G1", "0"))
    return c


def _bench_circuit(builder, *, toggle_sw_mid: bool = False) -> tuple[float, float, float]:
    """Return ``(steps_per_s_no_sink, steps_per_s_with_sink, overhead_pct)``."""
    times_no: list[float] = []
    times_yes: list[float] = []
    chunk_steps = 500
    n_chunks = 4  # 2,000 timed steps per trial

    for trial in range(5):
        for use_sink in ((False, True) if trial % 2 == 0 else (True, False)):
            c = builder()
            s = SimulationSession(c, default_dt=10e-6)
            scope: ScopeCore | None = None
            if use_sink:
                scope = ScopeCore()
                probe_node = [n for n in c.all_nodes() if n != "0"][0]
                scope.assign_probe("CH1", probe_node, enabled=True)
                scope.bind_session(s)

            s.start()
            s.step_chunk(50)  # warm-up

            gc.disable()
            try:
                t0 = time.perf_counter()
                for idx in range(n_chunks):
                    if toggle_sw_mid and idx == 2:
                        s.toggle_switch("SW1")
                    s.step_chunk(chunk_steps, dt=10e-6)
                elapsed = time.perf_counter() - t0
            finally:
                gc.enable()

            if use_sink:
                times_yes.append(elapsed)
            else:
                times_no.append(elapsed)

    med_no = statistics.median(times_no)
    med_yes = statistics.median(times_yes)
    total_steps = chunk_steps * n_chunks
    rate_no = total_steps / med_no
    rate_yes = total_steps / med_yes
    overhead_pct = ((med_yes - med_no) / med_no) * 100.0
    return rate_no, rate_yes, overhead_pct


def test_benchmark_scope_4ch_produce_frame_under_2ms() -> None:
    """4-channel 1000-pt `produce_frame()` with measurements, math and FFT runs quickly (< 8 ms hard limit)."""
    c = _build_circuit_a_rc()
    s = SimulationSession(c, default_dt=10e-6)
    scope = ScopeCore()
    scope.assign_probe("CH1", "in", enabled=True)
    scope.assign_probe("CH2", "n1", enabled=True)
    scope.assign_probe("CH3", "out", enabled=True)
    scope.assign_probe("CH4", "0", enabled=True)
    scope.set_math(True, MathOp.SUB, source_a="CH1", source_b="CH2")
    scope.set_fft("CH1", window="hann")
    scope.bind_session(s)
    scope.set_timebase(1e-3)
    s.step_chunk(1500)

    # Warm-up
    for _ in range(5):
        scope.produce_frame()

    t0 = time.perf_counter()
    iters = 30
    for _ in range(iters):
        frame = scope.produce_frame()
    avg_ms = ((time.perf_counter() - t0) / iters) * 1000.0
    print(f"\nBENCH_FRAME_4CH: avg_ms={avg_ms:.3f}, ch_count={len(frame.traces or {})}")
    assert len(frame.traces or {}) == 4
    assert avg_ms < 8.0, f"4ch produce_frame took {avg_ms:.3f} ms (expected < 8.0 ms)"


def test_benchmark_solver_sink_overhead_under_5pct() -> None:
    """Attached `ScopeCore` sample capture adds < 5% solver runtime overhead across circuits (a)-(d)."""
    specs = [
        ("(a) 5-part RC", _build_circuit_a_rc, False),
        ("(b) 30-R ladder + RC", _build_circuit_b_ladder, False),
        ("(c) RL + switch", _build_circuit_c_rl_switch, True),
        ("(d) LED + R", _build_circuit_d_led, False),
    ]
    for label, builder, tog in specs:
        rate_no, rate_yes, oh_pct = _bench_circuit(builder, toggle_sw_mid=tog)
        print(
            f"BENCH_SINK {label}: no_sink={rate_no:,.0f} steps/s, "
            f"with_sink={rate_yes:,.0f} steps/s, overhead={oh_pct:+.2f}%"
        )
        assert oh_pct < 5.0, f"{label} sink overhead {oh_pct:.2f}% >= 5%"


def test_benchmark_ui_responsiveness_ladder_with_scope_sink(qapp) -> None:
    """MainWindow UI loop keeps max timer gap < 50 ms on 30-resistor ladder + scope sink active."""
    from circuitsim.app.main_window import MainWindow

    win = MainWindow()
    scene = win.circuit_scene

    scene.place_new_component("ac_voltage", QPointF(0, 0), rotation_deg=90)
    scene.place_new_component("ground", QPointF(0, 60))
    for k in range(15):
        x_left = k * 80
        x_right = (k + 1) * 80
        scene.place_new_component("resistor", QPointF(x_left + 40, -40), rotation_deg=0)
        scene.place_new_component("resistor", QPointF(x_right, 0), rotation_deg=90)
        scene.add_wire([(x_left, 40), (x_right, 40)])
    scene.place_new_component("resistor", QPointF(1240, -40), rotation_deg=0)
    scene.place_new_component("capacitor", QPointF(1280, 0), rotation_deg=90)
    scene.add_wire([(1200, 40), (1280, 40)])

    scope = ScopeCore()
    scope.assign_probe("CH1", "n15", enabled=True)
    scope.set_timebase(1e-3)
    win.sim_controller._worker.session.attach_scope_sink(scope)

    def on_frame(_sim_frame) -> None:
        if scope.buffer.size >= 20:
            scope.produce_frame()

    win.sim_controller.frame_ready.connect(on_frame)

    gaps: list[float] = []
    last_tick = time.perf_counter()
    heartbeats = 0

    def on_heartbeat() -> None:
        nonlocal last_tick, heartbeats
        now = time.perf_counter()
        if heartbeats > 0:
            gaps.append((now - last_tick) * 1000.0)
        last_tick = now
        heartbeats += 1

    qapp.processEvents()
    hb = QTimer()
    hb.setInterval(10)
    hb.timeout.connect(on_heartbeat)
    hb.start()

    win.start_simulation()
    loop = QEventLoop()
    QTimer.singleShot(300, loop.quit)
    loop.exec()
    hb.stop()

    max_gap = max(gaps) if gaps else 0.0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    print(f"BENCH_UI_SCOPE: heartbeats={heartbeats}, avg_gap_ms={avg_gap:.2f}, max_gap_ms={max_gap:.2f}")
    assert heartbeats >= 12
    assert max_gap < 50.0, f"UI max gap {max_gap:.2f} ms >= 50.0 ms"

    win.circuit_scene.undo_stack.setClean()
    win.close()

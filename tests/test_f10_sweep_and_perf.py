"""Tests for Item F10: Parallel parameter sweep, seeded Monte Carlo, & 60+ part cache/perf."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import time
import numpy as np

from PySide6.QtCore import QPointF

from circuitsim.components import Capacitor, DCVoltageSource, Ground, Resistor
from circuitsim.engine import (
    Circuit,
    MNASolver,
    MonteCarloTolerance,
    run_monte_carlo,
    run_parameter_sweep,
)


def test_f10_sweep_module_is_qt_free() -> None:
    """Verify `circuitsim.engine.sweep` contains no Qt imports."""
    path = Path(__file__).resolve().parent.parent / "src" / "circuitsim" / "engine" / "sweep.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("PySide6", "PyQt")), alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("PySide6", "PyQt")), node.module


def test_f10_parallel_parameter_sweep_dc_and_transient() -> None:
    """Verify parallel DC and transient parameter sweeps preserve order and exact math."""
    c_div = Circuit()
    c_div.add(Ground("GND", "0"))
    c_div.add(DCVoltageSource("V1", "vin", "0", voltage=10.0))
    c_div.add(Resistor("R1", "vin", "vout", resistance=1000.0))
    c_div.add(Resistor("R2", "vout", "0", resistance=1000.0))

    r2_vals = [500.0, 1000.0, 2000.0, 4000.0, 9000.0]
    sw_dc = run_parameter_sweep(
        c_div,
        component_name="R2",
        parameter="resistance",
        values=r2_vals,
        mode="dc",
        probe_node="vout",
        max_workers=4,
    )
    for r2, v_sim in zip(r2_vals, sw_dc.node_voltages["vout"]):
        expected = 10.0 * r2 / (1000.0 + r2)
        assert math.isclose(float(v_sim), expected, rel_tol=1e-9)
    assert np.allclose(sw_dc.metrics, sw_dc.node_voltages["vout"], atol=1e-12)

    # Transient RC capacitance sweep at t = 1 ms
    c_rc = Circuit()
    c_rc.add(Ground("GND", "0"))
    c_rc.add(DCVoltageSource("V1", "vin", "0", voltage=10.0))
    c_rc.add(Resistor("R1", "vin", "vout", resistance=1000.0))
    c_rc.add(Capacitor("C1", "vout", "0", capacitance=1e-6))

    c_vals = [0.5e-6, 1.0e-6, 2.0e-6]
    sw_tr = run_parameter_sweep(
        c_rc,
        component_name="C1",
        parameter="capacitance",
        values=c_vals,
        mode="transient",
        duration=1e-3,
        dt=5e-6,
        probe_node="vout",
        max_workers=3,
    )
    for c_val, v_end in zip(c_vals, sw_tr.node_voltages["vout"]):
        expected = 10.0 * (1.0 - math.exp(-1e-3 / (1000.0 * c_val)))
        assert math.isclose(float(v_end), expected, rel_tol=2e-3)


def test_f10_seeded_monte_carlo_reproducible_and_statistics() -> None:
    """Verify seeded Monte Carlo is bit-identical across workers/runs and statistically sound."""
    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "vin", "0", voltage=10.0))
    c.add(Resistor("R1", "vin", "vout", resistance=10_000.0))
    c.add(Resistor("R2", "vout", "0", resistance=10_000.0))

    tols = [
        MonteCarloTolerance("R1", "resistance", tolerance_pct=5.0, distribution="uniform"),
        MonteCarloTolerance("R2", "resistance", tolerance_pct=5.0, distribution="uniform"),
    ]
    mc_a = run_monte_carlo(
        c, tolerances=tols, num_trials=60, seed=12345, probe_node="vout", max_workers=4
    )
    mc_b = run_monte_carlo(
        c, tolerances=tols, num_trials=60, seed=12345, probe_node="vout", max_workers=1
    )
    mc_other = run_monte_carlo(
        c, tolerances=tols, num_trials=60, seed=99999, probe_node="vout", max_workers=4
    )

    assert math.isclose(mc_a.nominal_metric, 5.0, rel_tol=1e-7)
    assert np.array_equal(mc_a.trial_metrics, mc_b.trial_metrics)
    assert not np.array_equal(mc_a.trial_metrics, mc_other.trial_metrics)
    assert 4.8 <= mc_a.mean <= 5.2
    assert mc_a.std > 0.0
    assert mc_a.min_val <= mc_a.p05 <= mc_a.p50 <= mc_a.p95 <= mc_a.max_val
    # With +/-5% uniform tolerances on R1, R2: worst-case range is [4.7368, 5.2632] V
    assert mc_a.min_val >= 10.0 * 0.95 / (1.05 + 0.95) - 1e-9
    assert mc_a.max_val <= 10.0 * 1.05 / (0.95 + 1.05) + 1e-9


def test_f10_large_65_part_circuit_perf_and_geometry_cache(qapp) -> None:
    """Verify 60+ component schematic solves quickly and hits WireItem/ComponentItem caches."""
    from circuitsim.canvas.scene import CircuitScene
    from circuitsim.canvas.symbol_renderer import SymbolLibrary
    from circuitsim.wiring.model import TerminalRef

    # 1. 65-part engine ladder solve
    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "n0", "0", voltage=64.0))
    for k in range(64):
        n_next = "0" if k == 63 else f"n{k + 1}"
        c.add(Resistor(f"R{k + 1}", f"n{k}", n_next, resistance=100.0))
    assert len(c.components) == 66

    t0 = time.perf_counter()
    dc = MNASolver(c).solve_dc()
    elapsed_solve = time.perf_counter() - t0
    assert elapsed_solve < 0.5
    assert math.isclose(dc.node_voltages["n32"], 32.0, rel_tol=1e-7)

    # 2. 62-component + 61-wire CircuitScene geometry cache check
    scene = CircuitScene(SymbolLibrary())
    g = scene.wire_graph
    v_item = scene.place_new_component("dc_voltage", QPointF(0.0, 0.0))
    r_items = []
    for k in range(60):
        row, col = divmod(k, 10)
        it = scene.place_new_component(
            "resistor", QPointF(160.0 * (col + 1), 120.0 * row)
        )
        r_items.append(it)
    gnd_item = scene.place_new_component("ground", QPointF(160.0 * 11, 0.0))
    assert len(scene.placed_items) == 62

    g = g.add_wire(TerminalRef(v_item.component.name, "pos"), TerminalRef(r_items[0].component.name, "a"))
    for k in range(59):
        g = g.add_wire(
            TerminalRef(r_items[k].component.name, "b"),
            TerminalRef(r_items[k + 1].component.name, "a"),
        )
    g = g.add_wire(TerminalRef(r_items[-1].component.name, "b"), TerminalRef(gnd_item.component.name, "gnd"))
    g = g.add_wire(TerminalRef(v_item.component.name, "neg"), TerminalRef(gnd_item.component.name, "gnd"))
    scene.set_wire_graph(g)
    assert len(scene.placed_wires) == 62

    # Repeated boundingRect() calls hit O(1) cache on ComponentItem & WireItem
    sample_comp = r_items[0]
    _ = sample_comp.boundingRect()
    base_count = sample_comp._bounding_rect_recompute_count
    for _ in range(25):
        _ = sample_comp.boundingRect()
    assert sample_comp._bounding_rect_recompute_count == base_count

    sample_wire = scene.placed_wires[0]
    assert sample_wire.boundingRect() is sample_wire._cached_bounding_rect

"""Qt-free parallel parameter sweeps and seeded Monte Carlo tolerance analyzer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field
import math
from typing import Callable, Iterable, Mapping, Sequence
import numpy as np

from circuitsim.components.capacitor import Capacitor
from circuitsim.components.inductor import Inductor
from circuitsim.components.potentiometer import Potentiometer
from circuitsim.components.resistor import Resistor
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.solver import IntegrationMethod, MNASolver, SolveResult, TransientResult


def _set_component_param(circuit: Circuit, comp_name: str, param: str, value: float) -> None:
    comp = circuit.get_component(comp_name)
    try:
        comp.set_parameter_value(param, float(value))
    except KeyError:
        if not hasattr(comp, param):
            raise
        setattr(comp, param, float(value))


@dataclass(frozen=True)
class ParameterSweepResult:
    """Ordered output from `run_parameter_sweep`."""

    component_name: str
    parameter: str
    values: np.ndarray
    mode: str
    probe_node: str | None
    metrics: np.ndarray
    node_voltages: dict[str, np.ndarray]
    results: tuple[SolveResult | TransientResult, ...]


def run_parameter_sweep(
    circuit: Circuit,
    *,
    component_name: str,
    parameter: str,
    values: Sequence[float],
    mode: str = "dc",
    duration: float = 1e-3,
    dt: float | None = None,
    method: IntegrationMethod | str = IntegrationMethod.TRAPEZOIDAL,
    probe_node: str | None = None,
    probe_nodes: Sequence[str] = (),
    metric_fn: Callable[[SolveResult | TransientResult], float] | None = None,
    max_workers: int = 4,
    regularize_ideal_loops: bool = False,
) -> ParameterSweepResult:
    """Evaluate `circuit` across `values` of `component_name.parameter` in parallel."""
    val_arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if val_arr.ndim != 1 or len(val_arr) == 0:
        raise ValueError("values must contain at least one numeric sweep point.")

    mode_norm = mode.strip().lower()
    if mode_norm not in ("dc", "transient"):
        raise ValueError(f"Unsupported sweep mode {mode!r}; expected 'dc' or 'transient'.")

    tracked_nodes = list(probe_nodes)
    if probe_node and probe_node not in tracked_nodes:
        tracked_nodes.insert(0, probe_node)

    def eval_point(val: float) -> tuple[SolveResult | TransientResult, float, dict[str, float]]:
        circ_copy = copy.deepcopy(circuit)
        _set_component_param(circ_copy, component_name, parameter, val)
        solver = MNASolver(circ_copy, regularize_ideal_loops=regularize_ideal_loops)
        if mode_norm == "dc":
            sol = solver.solve_dc()
            nv = {n: float(sol.node_voltages.get(n, 0.0)) for n in tracked_nodes}
            if metric_fn is not None:
                m = float(metric_fn(sol))
            elif probe_node is not None:
                m = float(sol.node_voltages.get(probe_node, 0.0))
            else:
                m = 0.0
            return sol, m, nv
        tr = solver.solve_transient(duration=duration, dt=dt, method=method)
        nv = {
            n: float(tr.node_voltages[n][-1]) if n in tr.node_voltages else 0.0
            for n in tracked_nodes
        }
        if metric_fn is not None:
            m = float(metric_fn(tr))
        elif probe_node is not None and probe_node in tr.node_voltages:
            m = float(tr.node_voltages[probe_node][-1])
        else:
            m = 0.0
        return tr, m, nv

    n_pts = len(val_arr)
    workers = max(1, min(int(max_workers), n_pts))
    if workers == 1:
        outputs = [eval_point(float(v)) for v in val_arr]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(eval_point, [float(v) for v in val_arr]))

    results_list = [o[0] for o in outputs]
    metrics_arr = np.array([o[1] for o in outputs], dtype=np.float64)
    node_arrays: dict[str, np.ndarray] = {
        n: np.array([o[2][n] for o in outputs], dtype=np.float64) for n in tracked_nodes
    }
    return ParameterSweepResult(
        component_name=component_name,
        parameter=parameter,
        values=val_arr,
        mode=mode_norm,
        probe_node=probe_node,
        metrics=metrics_arr,
        node_voltages=node_arrays,
        results=tuple(results_list),
    )


@dataclass(frozen=True)
class MonteCarloTolerance:
    """Component parameter tolerance specification for Monte Carlo trials."""

    component_name: str
    parameter: str
    rel_std: float | None = None
    abs_std: float | None = None
    tolerance_pct: float | None = None
    distribution: str = "normal"  # "normal" | "uniform"


@dataclass(frozen=True)
class MonteCarloResult:
    """Seeded Monte Carlo statistical summary across randomized component trials."""

    seed: int
    num_trials: int
    nominal_metric: float
    trial_metrics: np.ndarray
    mean: float
    std: float
    min_val: float
    max_val: float
    p05: float
    p50: float
    p95: float
    sampled_parameters: tuple[dict[tuple[str, str], float], ...] = ()


def _default_tolerances(
    circuit: Circuit,
    *,
    default_rel_std: float | None,
    default_tolerance_pct: float | None,
    distribution: str,
) -> list[MonteCarloTolerance]:
    tols: list[MonteCarloTolerance] = []
    for comp in circuit.components:
        if isinstance(comp, Resistor):
            param = "resistance"
        elif isinstance(comp, Capacitor):
            param = "capacitance"
        elif isinstance(comp, Inductor):
            param = "inductance"
        elif isinstance(comp, Potentiometer):
            param = "resistance"
        else:
            continue
        tols.append(
            MonteCarloTolerance(
                component_name=comp.name,
                parameter=param,
                rel_std=default_rel_std,
                tolerance_pct=default_tolerance_pct,
                distribution=distribution,
            )
        )
    return tols


def run_monte_carlo(
    circuit: Circuit,
    *,
    tolerances: Iterable[MonteCarloTolerance] | None = None,
    default_rel_std: float | None = None,
    default_tolerance_pct: float | None = 5.0,
    distribution: str = "normal",
    num_trials: int = 50,
    seed: int = 42,
    mode: str = "dc",
    duration: float = 1e-3,
    dt: float | None = None,
    method: IntegrationMethod | str = IntegrationMethod.TRAPEZOIDAL,
    probe_node: str | None = None,
    metric_fn: Callable[[SolveResult | TransientResult], float] | None = None,
    max_workers: int = 4,
    regularize_ideal_loops: bool = False,
) -> MonteCarloResult:
    """Run deterministic seeded Monte Carlo tolerance trials across parallel workers."""
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}.")

    mode_norm = mode.strip().lower()
    if mode_norm not in ("dc", "transient"):
        raise ValueError(f"Unsupported Monte Carlo mode {mode!r}; expected 'dc' or 'transient'.")

    tol_list = (
        list(tolerances)
        if tolerances is not None
        else _default_tolerances(
            circuit,
            default_rel_std=default_rel_std,
            default_tolerance_pct=default_tolerance_pct,
            distribution=distribution,
        )
    )

    def _run_one(circ_obj: Circuit) -> float:
        solver = MNASolver(circ_obj, regularize_ideal_loops=regularize_ideal_loops)
        if mode_norm == "dc":
            sol = solver.solve_dc()
            if metric_fn is not None:
                return float(metric_fn(sol))
            if probe_node is not None:
                return float(sol.node_voltages.get(probe_node, 0.0))
            return 0.0
        tr = solver.solve_transient(duration=duration, dt=dt, method=method)
        if metric_fn is not None:
            return float(metric_fn(tr))
        if probe_node is not None and probe_node in tr.node_voltages:
            return float(tr.node_voltages[probe_node][-1])
        return 0.0

    nominal_metric = _run_one(copy.deepcopy(circuit))

    # Pre-generate all random perturbations sequentially so seed reproducibility
    # is invariant to worker count or execution order.
    rng = np.random.default_rng(int(seed))
    nominal_vals: dict[tuple[str, str], float] = {}
    for spec in tol_list:
        comp = circuit.get_component(spec.component_name)
        nominal_vals[(spec.component_name, spec.parameter)] = float(
            getattr(comp, spec.parameter)
        )

    trial_params: list[dict[tuple[str, str], float]] = []
    for _ in range(num_trials):
        row: dict[tuple[str, str], float] = {}
        for spec in tol_list:
            nom = nominal_vals[(spec.component_name, spec.parameter)]
            dist = spec.distribution.strip().lower()
            if spec.abs_std is not None:
                sampled = nom + float(rng.normal(0.0, spec.abs_std))
            elif spec.rel_std is not None:
                factor = 1.0 + float(rng.normal(0.0, spec.rel_std))
                sampled = nom * max(factor, 1e-6)
            else:
                tol_frac = ((spec.tolerance_pct if spec.tolerance_pct is not None else 5.0) / 100.0)
                if dist == "uniform":
                    factor = float(rng.uniform(1.0 - tol_frac, 1.0 + tol_frac))
                else:
                    # 3-sigma Gaussian within stated tolerance_pct
                    factor = 1.0 + float(rng.normal(0.0, tol_frac / 3.0))
                sampled = nom * max(factor, 1e-6)
            row[(spec.component_name, spec.parameter)] = sampled
        trial_params.append(row)

    def eval_trial(p_map: dict[tuple[str, str], float]) -> float:
        c_copy = copy.deepcopy(circuit)
        for (cname, pname), val in p_map.items():
            _set_component_param(c_copy, cname, pname, val)
        return _run_one(c_copy)

    workers = max(1, min(int(max_workers), num_trials))
    if workers == 1:
        metrics = np.array([eval_trial(p) for p in trial_params], dtype=np.float64)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            metrics = np.array(list(pool.map(eval_trial, trial_params)), dtype=np.float64)

    return MonteCarloResult(
        seed=int(seed),
        num_trials=int(num_trials),
        nominal_metric=nominal_metric,
        trial_metrics=metrics,
        mean=float(np.mean(metrics)),
        std=float(np.std(metrics, ddof=1 if num_trials > 1 else 0)),
        min_val=float(np.min(metrics)),
        max_val=float(np.max(metrics)),
        p05=float(np.percentile(metrics, 5)),
        p50=float(np.percentile(metrics, 50)),
        p95=float(np.percentile(metrics, 95)),
        sampled_parameters=tuple(trial_params),
    )

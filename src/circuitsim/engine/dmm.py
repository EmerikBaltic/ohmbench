"""Qt-free Digital Multimeter (DMM) measurement engine (DC V, AC V RMS, DC A, Resistance/Continuity)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from circuitsim.components.sources import ACVoltageSource, WavVoltageSource
from circuitsim.components.ground import Ground
from circuitsim.dialogs.value_parser import format_engineering_value
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.solver import MNASolver, SolveResult, TransientResult

CONTINUITY_THRESHOLD_OHM: float = 50.0


@dataclass(frozen=True)
class DMMReading:
    """Result of a virtual Digital Multimeter measurement."""

    mode: str
    value: float
    unit: str
    formatted_text: str
    continuity_active: bool = False


def _resolve_probe_nodes(
    circuit: Circuit, pos_target: str, neg_target: str
) -> tuple[str, str, str | None]:
    """Return `(n_pos, n_neg, matched_comp_name)` supporting component names or node names."""
    try:
        comp = circuit.get_component(pos_target)
        if len(comp.nodes) >= 2:
            return (comp.nodes[0], comp.nodes[1], comp.name)
    except KeyError:
        pass
    return (pos_target, neg_target or "0", None)


def measure_dmm(
    circuit: Circuit,
    *,
    mode: str,
    pos_target: str,
    neg_target: str = "0",
    solve_result: SolveResult | None = None,
    transient_result: TransientResult | None = None,
) -> DMMReading:
    """Measure DC V, AC RMS V, DC A, or passive Resistance/Continuity on ``circuit``."""
    norm_mode = str(mode or "dc_v").strip().lower()
    n_pos, n_neg, comp_name = _resolve_probe_nodes(circuit, pos_target, neg_target)

    if norm_mode == "dc_v":
        res = solve_result if solve_result is not None else MNASolver(circuit).solve_dc()
        val = float(res.node_voltages.get(n_pos, 0.0) - res.node_voltages.get(n_neg, 0.0))
        return DMMReading(
            mode="dc_v",
            value=val,
            unit="V",
            formatted_text=format_engineering_value(val, "V"),
            continuity_active=False,
        )

    if norm_mode == "ac_v_rms":
        tr = transient_result
        if tr is None:
            freqs = [
                c.frequency
                for c in circuit.components
                if isinstance(c, ACVoltageSource) and c.frequency > 0.0
            ]
            if freqs:
                f_min = min(freqs)
                duration = max(2.0 / f_min, 1e-4)
                dt = min(duration / 400.0, 1.0 / (max(freqs) * 60.0))
            else:
                duration = 20e-3
                dt = 1e-4
            tr = MNASolver(circuit).solve_transient(duration=duration, dt=dt)
        n_samples = len(tr.times)
        vp = tr.node_voltages.get(n_pos, np.zeros(n_samples, dtype=np.float64))
        vn = tr.node_voltages.get(n_neg, np.zeros(n_samples, dtype=np.float64))
        diff = np.asarray(vp, dtype=np.float64) - np.asarray(vn, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(diff)))) if len(diff) > 0 else 0.0
        return DMMReading(
            mode="ac_v_rms",
            value=rms,
            unit="V",
            formatted_text=f"{format_engineering_value(rms, 'V')} RMS",
            continuity_active=False,
        )

    if norm_mode == "dc_a":
        res = solve_result if solve_result is not None else MNASolver(circuit).solve_dc()
        key = comp_name or pos_target
        i_val = float(res.branch_currents.get(key, 0.0))
        return DMMReading(
            mode="dc_a",
            value=i_val,
            unit="A",
            formatted_text=format_engineering_value(i_val, "A"),
            continuity_active=False,
        )

    if norm_mode == "resistance":
        probe_circ = circuit
        if not circuit.grounded_nodes():
            probe_circ = Circuit()
            for comp in circuit.components:
                probe_circ.add(comp)
            probe_circ.add(Ground("__dmm_gnd", n_neg))
        r_eq = float(probe_circ.thevenin_resistance(n_pos, n_neg))
        if not math.isfinite(r_eq):
            return DMMReading(
                mode="resistance",
                value=float("inf"),
                unit="Ω",
                formatted_text="OL (Open)",
                continuity_active=False,
            )
        cont = 0.0 <= r_eq < CONTINUITY_THRESHOLD_OHM
        txt = format_engineering_value(r_eq, "Ω")
        if cont:
            txt = f"{txt} [CONT]"
        return DMMReading(
            mode="resistance",
            value=r_eq,
            unit="Ω",
            formatted_text=txt,
            continuity_active=cont,
        )

    raise ValueError(
        f"Unsupported DMM mode {mode!r}; expected 'dc_v', 'ac_v_rms', 'dc_a', or 'resistance'."
    )

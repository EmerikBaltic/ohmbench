"""Qt-free component stress, derating, and overload/reverse-polarity analyzer (Item F4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


class StressLevel(str, Enum):
    """Component electrical stress classification."""

    NORMAL = "normal"
    WARNING = "warning"
    OVERLOAD = "overload"


@dataclass(frozen=True)
class StressReport:
    """Stress and derating assessment for a single component."""

    component_name: str
    level: StressLevel
    utilization_pct: float
    reason: str
    metric_name: str = "power"


def _classify_utilization(
    comp_name: str,
    utilization_pct: float,
    metric_name: str,
    value_desc: str,
    limit_desc: str,
) -> StressReport:
    if not math.isfinite(utilization_pct):
        return StressReport(
            component_name=comp_name,
            level=StressLevel.OVERLOAD,
            utilization_pct=float("inf"),
            reason=f"{metric_name} non-finite ({value_desc} vs {limit_desc})",
            metric_name=metric_name,
        )
    if utilization_pct > 100.0 + 1e-6:
        return StressReport(
            component_name=comp_name,
            level=StressLevel.OVERLOAD,
            utilization_pct=utilization_pct,
            reason=f"{metric_name} overload: {value_desc} exceeds {limit_desc} ({utilization_pct:.1f}%)",
            metric_name=metric_name,
        )
    if utilization_pct >= 80.0 - 1e-6:
        return StressReport(
            component_name=comp_name,
            level=StressLevel.WARNING,
            utilization_pct=utilization_pct,
            reason=f"Derating warning: {value_desc} at {utilization_pct:.1f}% of {limit_desc}",
            metric_name=metric_name,
        )
    return StressReport(
        component_name=comp_name,
        level=StressLevel.NORMAL,
        utilization_pct=utilization_pct,
        reason=f"Normal: {utilization_pct:.1f}% of {limit_desc}",
        metric_name=metric_name,
    )


def _extract_component_records(
    graph_or_circuit: Any,
    solution: Any | None,
    components_by_name: Mapping[str, Any] | None = None,
) -> list[tuple[str, str, Any | None, float, float, float]]:
    """Normalize `(Circuit | WireGraph, SolveResult | SimulationFrame | None)` inputs."""
    comp_map: dict[str, Any] = dict(components_by_name or {})
    src_obj = getattr(graph_or_circuit, "circuit", graph_or_circuit)
    comps_attr = getattr(src_obj, "components", None)
    if isinstance(comps_attr, Mapping):
        for k, v in comps_attr.items():
            comp_map.setdefault(str(k), v)
    elif comps_attr is not None:
        for item in comps_attr:
            cname = getattr(item, "name", None)
            if cname is not None:
                comp_map.setdefault(str(cname), item)

    telem_map: Mapping[str, Any] = {}
    branch_currents: Mapping[str, float] = {}
    node_voltages: Mapping[str, float] = {}
    if solution is not None:
        if hasattr(solution, "components") and isinstance(
            getattr(solution, "components"), Mapping
        ):
            telem_map = solution.components
        if hasattr(solution, "branch_currents") and isinstance(
            getattr(solution, "branch_currents"), Mapping
        ):
            branch_currents = solution.branch_currents
        if hasattr(solution, "node_voltages") and isinstance(
            getattr(solution, "node_voltages"), Mapping
        ):
            node_voltages = solution.node_voltages

    names: list[str] = list(comp_map.keys())
    for k in telem_map:
        if k not in comp_map:
            names.append(str(k))

    records: list[tuple[str, str, Any | None, float, float, float]] = []
    for name in names:
        comp = comp_map.get(name)
        telem = telem_map.get(name)
        ctype = (
            getattr(comp, "component_type", "")
            if comp is not None
            else str(getattr(telem, "component_type", ""))
        )
        if telem is not None:
            v = float(getattr(telem, "voltage", 0.0))
            i = float(getattr(telem, "current", 0.0))
            p = float(getattr(telem, "power", v * i))
        elif comp is not None:
            nodes_list = getattr(comp, "nodes", ())
            if node_voltages and len(nodes_list) == 2:
                va = float(node_voltages.get(nodes_list[0], 0.0))
                vb = float(node_voltages.get(nodes_list[1], 0.0))
                v = va - vb
                i = float(branch_currents.get(name, getattr(comp, "current", 0.0)))
                p = v * i if (branch_currents and name in branch_currents) else float(getattr(comp, "power", v * i))
            else:
                v = float(getattr(comp, "voltage", 0.0))
                i = float(branch_currents.get(name, getattr(comp, "current", 0.0)))
                p = float(getattr(comp, "power", v * i))
        else:
            continue
        records.append((name, ctype, comp, v, i, p))
    return records


def evaluate_component_stress(
    graph: Any,
    solution: Any | None = None,
    *,
    components: Mapping[str, Any] | None = None,
) -> dict[str, StressReport]:
    """Evaluate electrical stress and derating levels across circuit components.

    Checks:
      - Resistor / Potentiometer: `P = I^2 * R` (or `|P|`) vs `max_power_w` (default `0.25 W`).
      - LED / Diode: forward current vs `max_current_a` (default `0.030 A` for LED) and
        reverse voltage vs `max_reverse_v` (default `5.0 V` for LED).
      - Capacitor: reverse polarity check (when `polarized=True`) and voltage vs `voltage_rating`.
      - Lightbulb: power vs `rated_power` / voltage vs `rated_voltage`.
    """
    reports: dict[str, StressReport] = {}
    records = _extract_component_records(graph, solution, components)
    for name, ctype, comp, v_val, i_val, p_val in records:
        if ctype in ("resistor", "potentiometer"):
            max_p = float(
                getattr(
                    comp,
                    "max_power_w",
                    getattr(comp, "max_power", getattr(comp, "power_rating", 0.25)),
                )
            )
            max_p = max(max_p, 1e-12)
            r_val = getattr(comp, "resistance", None) if ctype == "resistor" else None
            p_diss = (
                abs(i_val) * abs(i_val) * float(r_val)
                if isinstance(r_val, (int, float))
                else abs(p_val)
            )
            util = (p_diss / max_p) * 100.0
            reports[name] = _classify_utilization(
                name,
                util,
                "power",
                f"{p_diss * 1000.0:.2f} mW" if p_diss < 1.0 else f"{p_diss:.3f} W",
                f"{max_p * 1000.0:.1f} mW" if max_p < 1.0 else f"{max_p:.3f} W",
            )
            continue

        if ctype in ("led", "diode"):
            i_max = float(getattr(comp, "max_current_a", getattr(comp, "max_current", 0.030)))
            i_max = max(i_max, 1e-12)
            v_rev_max = float(getattr(comp, "max_reverse_v", 5.0))
            v_rev_max = max(v_rev_max, 1e-12)
            # Check reverse voltage first if reverse-biased beyond rating
            if v_val < -v_rev_max:
                rev_util = (abs(v_val) / v_rev_max) * 100.0
                reports[name] = StressReport(
                    component_name=name,
                    level=StressLevel.OVERLOAD,
                    utilization_pct=rev_util,
                    reason=f"Reverse voltage {abs(v_val):.2f} V exceeds limit {v_rev_max:.2f} V",
                    metric_name="reverse_voltage",
                )
                continue
            i_fwd = max(0.0, i_val)
            i_util = (i_fwd / i_max) * 100.0
            v_rev_util = (max(0.0, -v_val) / v_rev_max) * 100.0
            if v_rev_util > i_util:
                reports[name] = _classify_utilization(
                    name,
                    v_rev_util,
                    "reverse_voltage",
                    f"{abs(v_val):.2f} V",
                    f"{v_rev_max:.2f} V",
                )
            else:
                reports[name] = _classify_utilization(
                    name,
                    i_util,
                    "current",
                    f"{i_fwd * 1000.0:.2f} mA",
                    f"{i_max * 1000.0:.1f} mA",
                )
            continue

        if ctype == "capacitor":
            polarized = bool(getattr(comp, "polarized", False))
            if polarized and v_val < -0.5:
                v_max = max(float(getattr(comp, "voltage_rating", 50.0)), 1e-12)
                reports[name] = StressReport(
                    component_name=name,
                    level=StressLevel.OVERLOAD,
                    utilization_pct=max(100.1, (abs(v_val) / v_max) * 100.0),
                    reason=f"Reverse polarity across polarized capacitor ({v_val:.2f} V)",
                    metric_name="reverse_voltage",
                )
                continue
            v_max = max(float(getattr(comp, "voltage_rating", getattr(comp, "max_voltage", 50.0))), 1e-12)
            v_util = (abs(v_val) / v_max) * 100.0
            reports[name] = _classify_utilization(
                name,
                v_util,
                "voltage",
                f"{abs(v_val):.2f} V",
                f"{v_max:.1f} V",
            )
            continue

        if ctype in ("zener_diode", "npn_bjt", "pnp_bjt", "nmos", "pmos"):
            max_p = max(
                float(getattr(comp, "max_power_w", getattr(comp, "max_power", 0.5))),
                1e-12,
            )
            p_diss = abs(p_val)
            util = (p_diss / max_p) * 100.0
            reports[name] = _classify_utilization(
                name,
                util,
                "power",
                f"{p_diss * 1000.0:.2f} mW" if p_diss < 1.0 else f"{p_diss:.3f} W",
                f"{max_p * 1000.0:.1f} mW" if max_p < 1.0 else f"{max_p:.3f} W",
            )
            continue

        if ctype == "lightbulb":
            p_max = max(float(getattr(comp, "rated_power", 0.5)), 1e-12)
            util = (abs(p_val) / p_max) * 100.0
            reports[name] = _classify_utilization(
                name,
                util,
                "power",
                f"{abs(p_val):.3f} W",
                f"{p_max:.2f} W",
            )
            continue

        # Default normal report for sources / meters / switches / ground
        reports[name] = StressReport(
            component_name=name,
            level=StressLevel.NORMAL,
            utilization_pct=0.0,
            reason="Normal",
            metric_name="power",
        )

    return reports

"""Qt-free wiring current estimation and visual overlay helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from circuitsim.engine.overlay import (
    compute_current_dot_velocity,
    format_power_probe_text,
    format_watts_si,
    map_voltage_to_rgb,
)
from circuitsim.wiring.model import FreeEnd, JunctionRef, TerminalRef, WireGraph


def _wire_node_key(att: Any) -> tuple[str, Any]:
    if isinstance(att, TerminalRef):
        return ("T", (att.component, att.terminal))
    if isinstance(att, JunctionRef):
        return ("J", att.id)
    if isinstance(att, FreeEnd):
        return ("F", (att.x, att.y))
    return ("X", att)


def estimate_wire_currents(
    graph: WireGraph,
    terminal_names_by_comp: Mapping[str, Sequence[str]],
    branch_currents: Mapping[str, float],
) -> dict[str, float]:
    """Estimate signed current (A) flowing `end_a -> end_b` along each wire via KCL leaf peeling.

    For 2-terminal components, solver passive sign convention defines positive `I_comp`
    as entering pin 0 and exiting pin 1.
    """
    adj: dict[tuple[str, Any], list[tuple[str, tuple[str, Any], int]]] = {}
    for w in graph.wires:
        ka = _wire_node_key(w.end_a)
        kb = _wire_node_key(w.end_b)
        adj.setdefault(ka, []).append((w.id, kb, +1))
        adj.setdefault(kb, []).append((w.id, ka, -1))

    injection: dict[tuple[str, Any], float] = {k: 0.0 for k in adj}
    known_node: dict[tuple[str, Any], bool] = {}
    for k in adj:
        kind, payload = k
        if kind == "F":
            injection[k] = 0.0
            known_node[k] = True
        elif kind == "J":
            injection[k] = 0.0
            known_node[k] = True
        elif kind == "T":
            cname, tname = payload
            tlist = terminal_names_by_comp.get(cname, ())
            i_comp = float(branch_currents.get(cname, 0.0))
            if len(tlist) == 2:
                if tname == tlist[0]:
                    injection[k] = -i_comp
                    known_node[k] = True
                elif tname == tlist[1]:
                    injection[k] = +i_comp
                    known_node[k] = True
                else:
                    known_node[k] = False
            else:
                known_node[k] = False
        else:
            known_node[k] = False

    remaining_wires: dict[tuple[str, Any], dict[str, tuple[tuple[str, Any], int]]] = {
        k: {wid: (nbr, sign) for wid, nbr, sign in edges} for k, edges in adj.items()
    }
    wire_currents: dict[str, float] = {w.id: 0.0 for w in graph.wires}
    resolved_wires: set[str] = set()

    changed = True
    while changed:
        changed = False
        for node, wmap in list(remaining_wires.items()):
            if not known_node.get(node, False) or len(wmap) != 1:
                continue
            wid, (nbr, sign) = next(iter(wmap.items()))
            if wid in resolved_wires:
                continue
            i_from_node = injection[node]
            wire_currents[wid] = i_from_node if sign > 0 else -i_from_node
            injection[nbr] = injection.get(nbr, 0.0) + i_from_node
            del wmap[wid]
            if wid in remaining_wires.get(nbr, {}):
                del remaining_wires[nbr][wid]
            resolved_wires.add(wid)
            changed = True

    return wire_currents


__all__ = [
    "compute_current_dot_velocity",
    "estimate_wire_currents",
    "format_power_probe_text",
    "format_watts_si",
    "map_voltage_to_rgb",
]

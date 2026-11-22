"""Netlist from a `WireGraph`: union-find over terminals and junctions (no Qt).

Electrical connectivity here is decided ONLY by which attachments a wire joins:

* a wire joins its two ends if both are `TerminalRef`/`JunctionRef`,
* a `FreeEnd` connects to nothing (even if it lies exactly on a terminal),
* coordinates are never looked at, so crossing or overlapping wires never connect.

Union-find in one paragraph: start with every terminal and junction in its own
group; for every wire, merge the groups of its two ends; at the end each group is
one electrical node (net).

Net names follow the old extractor: the group containing a Ground terminal is
"0"; every other group that contains at least one part terminal is "N1", "N2", ...
Numbering is by the first part terminal of each group in natural order of name
(R2 before R10), so it does not depend on where things are drawn. (The old
extractor numbered by screen position, so N-numbers can differ; only the naming
scheme and "0" are the same.)
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from circuitsim.canvas.symbol_geometry import (
    DEFAULT_IEC_SYMBOLS_DIR,
    SymbolMetadata,
    load_all_iec_metadata,
)
from circuitsim.components.base import Component
from circuitsim.components.ground import Ground
from circuitsim.engine.netlist import Circuit
from circuitsim.wiring.model import (
    Attachment,
    FreeEnd,
    JunctionRef,
    TerminalRef,
    WireGraph,
    WireGraphError,
)

GROUND_NODE: str = "0"


class _UnionFind:
    """Disjoint sets over hashable keys (path compression)."""

    def __init__(self) -> None:
        self._parent: dict[object, object] = {}

    def find(self, item: object) -> object:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while item != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: object, b: object) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _natural_key(text: str) -> tuple[object, ...]:
    """'R10' sorts after 'R2' (digit runs compare as numbers)."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text))


@dataclass
class WiringNetlist:
    """Result of building a circuit from a `WireGraph` (mirrors the old `ConnectivityResult`)."""

    circuit: Circuit
    component_nodes: dict[str, dict[str, str]] = field(default_factory=dict)
    wire_to_node: dict[str, str] = field(default_factory=dict)
    junction_to_node: dict[str, str] = field(default_factory=dict)
    unconnected_components: tuple[str, ...] = ()
    nets: dict[str, tuple[TerminalRef, ...]] = field(default_factory=dict)

    def partition(self) -> frozenset[frozenset[TerminalRef]]:
        """The nets as a set of terminal sets (independent of the N-numbers)."""
        return frozenset(frozenset(ts) for ts in self.nets.values())


def terminal_names_by_component(
    components: Sequence[Component], catalog: Mapping[str, SymbolMetadata]
) -> dict[str, tuple[str, ...]]:
    """`{component name: terminal names in catalog order}`; rejects duplicate names."""
    result: dict[str, tuple[str, ...]] = {}
    for comp in components:
        if comp.name in result:
            raise WireGraphError(f"Two components are named {comp.name}.")
        if hasattr(comp, "macro_def"):
            result[comp.name] = tuple(comp.macro_def.pins)  # type: ignore[attr-defined]
        else:
            result[comp.name] = tuple(t.name for t in catalog[comp.component_type].terminals)
    return result


def _is_ground(comp: Component) -> bool:
    return isinstance(comp, Ground) or comp.component_type == "ground"


def build_circuit(
    graph: WireGraph,
    components: Sequence[Component],
    *,
    symbol_catalog: Mapping[str, SymbolMetadata] | None = None,
) -> WiringNetlist:
    """Build a solver-ready `Circuit` from components and the wire graph.

    Raises `WireGraphError` if the graph attaches to a missing terminal/junction.
    Parts with no wire on any terminal are left out of the circuit (as before) so
    that one stray part does not stop the rest from simulating.
    """
    catalog = (
        symbol_catalog
        if symbol_catalog is not None
        else load_all_iec_metadata(DEFAULT_IEC_SYMBOLS_DIR, include_extensions=True)
    )
    terminals = terminal_names_by_component(components, catalog)
    graph.validate(terminals)

    uf = _UnionFind()
    for comp in components:
        for tname in terminals[comp.name]:
            uf.find(TerminalRef(comp.name, tname))
    for junc in graph.junctions:
        uf.find(JunctionRef(junc.id))

    def key(att: Attachment) -> object | None:
        return None if isinstance(att, FreeEnd) else att

    wired_terminals: set[TerminalRef] = set()
    label_keys: dict[str, list[object]] = {}
    for w in graph.wires:
        ka, kb = key(w.end_a), key(w.end_b)
        for att in (w.end_a, w.end_b):
            if isinstance(att, TerminalRef):
                wired_terminals.add(att)
        if ka is not None and kb is not None:
            uf.union(ka, kb)
        norm_lbl = str(getattr(w, "net_label", "") or "").strip().upper()
        if norm_lbl:
            rep = ka if ka is not None else kb
            if rep is not None:
                label_keys.setdefault(norm_lbl, []).append(rep)

    for junc in graph.junctions:
        norm_lbl = str(getattr(junc, "net_label", "") or "").strip().upper()
        if norm_lbl:
            label_keys.setdefault(norm_lbl, []).append(JunctionRef(junc.id))

    for norm_lbl, keys in label_keys.items():
        if len(keys) >= 2:
            first_k = keys[0]
            for other_k in keys[1:]:
                uf.union(first_k, other_k)

    root_labels: dict[object, set[str]] = {}
    for norm_lbl, keys in label_keys.items():
        for k in keys:
            root_labels.setdefault(uf.find(k), set()).add(norm_lbl)

    # Group terminals by net
    groups: dict[object, list[TerminalRef]] = {}
    for comp in components:
        for tname in terminals[comp.name]:
            ref = TerminalRef(comp.name, tname)
            groups.setdefault(uf.find(ref), []).append(ref)

    ground_names = {c.name for c in components if _is_ground(c)}
    root_to_node: dict[object, str] = {}
    used_node_names: set[str] = {GROUND_NODE}
    others: list[tuple[tuple[object, ...], object]] = []
    for root, members in groups.items():
        lbls = root_labels.get(root, set())
        if any(m.component in ground_names for m in members) or any(
            lbl in (GROUND_NODE, "GND") for lbl in lbls
        ):
            root_to_node[root] = GROUND_NODE
        elif lbls:
            chosen = sorted(lbls)[0]
            root_to_node[root] = chosen
            used_node_names.add(chosen)
        else:
            first = min(members, key=lambda m: (_natural_key(m.component), m.terminal))
            others.append(((_natural_key(first.component), first.terminal), root))
    others.sort(key=lambda item: item[0])
    next_idx = 1
    for _, root in others:
        while f"N{next_idx}" in used_node_names:
            next_idx += 1
        node_name = f"N{next_idx}"
        next_idx += 1
        used_node_names.add(node_name)
        root_to_node[root] = node_name

    # Parts without any wire are ignored (unless nothing else is connected)
    def is_unconnected(comp: Component) -> bool:
        return not any(TerminalRef(comp.name, t) in wired_terminals for t in terminals[comp.name])

    unconnected = [c.name for c in components if is_unconnected(c)]
    has_connected_non_ground = any(
        not is_unconnected(c) and c.name not in ground_names for c in components
    )

    circuit = Circuit()
    component_nodes: dict[str, dict[str, str]] = {}
    for comp in components:
        term_map = {t: root_to_node[uf.find(TerminalRef(comp.name, t))] for t in terminals[comp.name]}
        component_nodes[comp.name] = term_map
        if is_unconnected(comp) and (has_connected_non_ground or comp.name in ground_names):
            continue
        clone = copy.deepcopy(comp)
        if hasattr(clone, "_source_mcu"):
            clone._source_mcu = comp
        if _is_ground(comp):
            clone.nodes = (GROUND_NODE,)
        else:
            clone.nodes = tuple(term_map[t] for t in terminals[comp.name])
        circuit.add(clone)

    wire_to_node: dict[str, str] = {}
    for w in graph.wires:
        for att in (w.end_a, w.end_b):
            if not isinstance(att, FreeEnd):
                root = uf.find(att)
                if root in root_to_node:
                    wire_to_node[w.id] = root_to_node[root]
                break
    junction_to_node = {
        j.id: root_to_node[uf.find(JunctionRef(j.id))]
        for j in graph.junctions
        if uf.find(JunctionRef(j.id)) in root_to_node
    }
    nets: dict[str, list[TerminalRef]] = {}
    for root, members in groups.items():
        nets[root_to_node[root]] = list(members)
    return WiringNetlist(
        circuit=circuit,
        component_nodes=component_nodes,
        wire_to_node=wire_to_node,
        junction_to_node=junction_to_node,
        unconnected_components=tuple(unconnected),
        nets={name: tuple(ts) for name, ts in nets.items()},
    )

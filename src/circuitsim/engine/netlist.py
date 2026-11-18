"""Circuit container, topological validator, and Thévenin resistance calculator."""

from __future__ import annotations

from collections import defaultdict, deque
import numpy as np

from circuitsim.components.base import Component
from circuitsim.components.bjt import BJTBase
from circuitsim.components.capacitor import Capacitor
from circuitsim.components.ground import Ground
from circuitsim.components.inductor import Inductor
from circuitsim.components.led import LED
from circuitsim.components.lightbulb import Lightbulb
from circuitsim.components.logic_gate import LogicGate
from circuitsim.components.mcu import MCUBlock
from circuitsim.components.meters import Ammeter, Voltmeter
from circuitsim.components.potentiometer import Potentiometer
from circuitsim.components.resistor import Resistor
from circuitsim.components.sources import ACVoltageSource, DCCurrentSource, DCVoltageSource
from circuitsim.components.switch import Switch
from circuitsim.engine.errors import (
    CircuitWarning,
    FloatingNodeError,
    NoGroundError,
    VoltageSourceLoopError,
    ZeroDelayLoopError,
)


class _DisjointSet:
    """Simple Union-Find data structure for cycle and connectivity checks."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self._parent:
            self._parent[item] = item
            return item
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        curr = item
        while curr != root:
            nxt = self._parent[curr]
            self._parent[curr] = root
            curr = nxt
        return root

    def union(self, a: str, b: str) -> bool:
        """Merge sets containing `a` and `b`. Returns False if already in same set."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        self._parent[root_b] = root_a
        return True


class Circuit:
    """Represents an electrical circuit composed of interconnected components.

    Responsibilities:
        - Stores components by unique name.
        - Tracks which nodes are grounded (0 V reference).
        - Validates circuit topology before solving (missing ground, floating
          nodes, ideal voltage-source loops, and ammeters in parallel with
          voltage sources).
        - Computes exact DC Thévenin resistance between any two nodes with all
          independent sources zeroed.
    """

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}
        self.version: int = 0

    def add(self, component: Component) -> Component:
        """Add a component to the circuit and increment the topology version."""
        if component.name in self._components:
            raise ValueError(f"Duplicate component name {component.name!r} in circuit.")
        self._components[component.name] = component
        self.version += 1
        return component

    def remove(self, name: str) -> Component:
        """Remove and return a component by name."""
        if name not in self._components:
            raise KeyError(f"Component {name!r} not found in circuit.")
        comp = self._components.pop(name)
        self.version += 1
        return comp

    def get_component(self, name: str) -> Component:
        """Return a component by its unique name."""
        if name not in self._components:
            raise KeyError(f"Component {name!r} not found in circuit.")
        return self._components[name]

    @property
    def components(self) -> list[Component]:
        """Return all components in insertion order."""
        return list(self._components.values())

    def mark_modified(self) -> None:
        """Mark the circuit parameters as modified so the solver updates its matrix."""
        self.version += 1

    def reset_state(self) -> None:
        """Reset all component states (e.g. switches, thermal filaments, meters)."""
        for comp in self._components.values():
            comp.reset_state()
        self.version += 1

    def grounded_nodes(self) -> set[str]:
        """Return the set of node names explicitly connected to a Ground component."""
        return {
            comp.node
            for comp in self._components.values()
            if isinstance(comp, Ground)
        }

    def all_nodes(self) -> set[str]:
        """Return all node names referenced by any component in the circuit."""
        nodes: set[str] = set()
        for comp in self._components.values():
            nodes.update(comp.nodes)
        return nodes

    def non_ground_nodes(self) -> list[str]:
        """Return sorted list of independent (non-grounded) node names."""
        gnd = self.grounded_nodes()
        return sorted(n for n in self.all_nodes() if n not in gnd)

    def thevenin_resistance(
        self,
        node_a: str,
        node_b: str,
        *,
        exclude_component: str | None = None,
    ) -> float:
        """Compute the DC Thévenin resistance between `node_a` and `node_b`.

        All independent voltage sources are zeroed (replaced by their `r_internal`,
        or a short circuit if `r_internal == 0`), independent current sources are
        opened (0 A), and capacitors are treated as open circuits.

        Args:
            node_a: First terminal node name.
            node_b: Second terminal node name.
            exclude_component: Optional component name (such as the target inductor
                itself) to exclude when measuring the external network resistance.
        """
        gnd_nodes = self.grounded_nodes()
        if node_a == node_b or (node_a in gnd_nodes and node_b in gnd_nodes):
            return 0.0

        non_gnd = self.non_ground_nodes()
        node_to_idx = {n: i for i, n in enumerate(non_gnd)}

        # Voltage sources become 0 V constraints with series r_internal.
        # Any other inductor with dcr == 0 also acts as a 0 V DC short branch.
        zeroed_vs_branches: list[tuple[str, str, float]] = []
        for comp in self._components.values():
            if comp.name == exclude_component or isinstance(comp, (Ground, Capacitor)):
                continue
            if isinstance(comp, (DCVoltageSource, ACVoltageSource)):
                zeroed_vs_branches.append((comp.nodes[0], comp.nodes[1], comp.r_internal))
            elif isinstance(comp, Inductor) and comp.dcr == 0.0:
                zeroed_vs_branches.append((comp.nodes[0], comp.nodes[1], 0.0))
            elif isinstance(comp, Ammeter) and getattr(comp, "is_ideal", False):
                zeroed_vs_branches.append((comp.nodes[0], comp.nodes[1], 0.0))

        n_nodes = len(non_gnd)
        n_vs = len(zeroed_vs_branches)
        dim = n_nodes + n_vs
        if dim == 0:
            return 0.0

        def build_matrix(gmin_val: float) -> np.ndarray:
            mat = np.zeros((dim, dim), dtype=np.float64)
            for idx in range(n_nodes):
                mat[idx, idx] += gmin_val

            for comp in self._components.values():
                if comp.name == exclude_component or isinstance(
                    comp, (Ground, Capacitor, DCVoltageSource, ACVoltageSource)
                ):
                    continue
                if isinstance(comp, Ammeter) and getattr(comp, "is_ideal", False):
                    continue
                if isinstance(comp, LogicGate):
                    for in_node in comp.input_nodes:
                        ii = node_to_idx.get(in_node)
                        if ii is not None:
                            mat[ii, ii] += comp.g_in
                    io = node_to_idx.get(comp.output_node)
                    if io is not None:
                        mat[io, io] += comp.g_out
                    continue
                if isinstance(comp, MCUBlock):
                    for pin, p_node in zip(comp.pin_names, comp.nodes):
                        ip = node_to_idx.get(p_node)
                        if ip is not None:
                            mat[ip, ip] += comp.pin_conductance(pin)
                    continue

                if isinstance(comp, Potentiometer):
                    for na, nb, g in (
                        (comp.node_a, comp.node_w, comp.g_aw),
                        (comp.node_w, comp.node_b, comp.g_wb),
                    ):
                        ia = node_to_idx.get(na)
                        ib = node_to_idx.get(nb)
                        if ia is not None:
                            mat[ia, ia] += g
                        if ib is not None:
                            mat[ib, ib] += g
                        if ia is not None and ib is not None:
                            mat[ia, ib] -= g
                            mat[ib, ia] -= g
                    continue

                g = 0.0
                if isinstance(comp, (Resistor, Switch, Lightbulb, Voltmeter, Ammeter, LED)):
                    g = comp.conductance
                elif isinstance(comp, Inductor) and comp.dcr > 0.0:
                    g = 1.0 / comp.dcr

                if g > 0.0:
                    ia = node_to_idx.get(comp.nodes[0])
                    ib = node_to_idx.get(comp.nodes[1])
                    if ia is not None:
                        mat[ia, ia] += g
                    if ib is not None:
                        mat[ib, ib] += g
                    if ia is not None and ib is not None:
                        mat[ia, ib] -= g
                        mat[ib, ia] -= g

            for k, (na, nb, r_int) in enumerate(zeroed_vs_branches):
                row_k = n_nodes + k
                ia = node_to_idx.get(na)
                ib = node_to_idx.get(nb)
                if ia is not None:
                    mat[ia, row_k] += 1.0
                    mat[row_k, ia] += 1.0
                if ib is not None:
                    mat[ib, row_k] -= 1.0
                    mat[row_k, ib] -= 1.0
                if r_int > 0.0:
                    mat[row_k, row_k] += r_int

            return mat

        rhs = np.zeros(dim, dtype=np.float64)
        idx_a = node_to_idx.get(node_a)
        idx_b = node_to_idx.get(node_b)
        if idx_a is not None:
            rhs[idx_a] += 1.0
        if idx_b is not None:
            rhs[idx_b] -= 1.0

        try:
            sol = np.linalg.solve(build_matrix(0.0), rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.solve(build_matrix(1e-12), rhs)

        va = float(sol[idx_a]) if idx_a is not None else 0.0
        vb = float(sol[idx_b]) if idx_b is not None else 0.0
        return max(0.0, va - vb)

    def is_ammeter_in_parallel(self, comp: Ammeter) -> bool:
        """Return True if comp is connected in parallel across a voltage source or short."""
        if not isinstance(comp, Ammeter) or len(comp.nodes) < 2:
            return False
        gnd_nodes = self.grounded_nodes()

        def canonical(node: str) -> str:
            return "__GND__" if node in gnd_nodes else node

        c_a = canonical(comp.nodes[0])
        c_b = canonical(comp.nodes[1])
        if c_a == c_b:
            return True

        battery_dsu = _DisjointSet()
        for c in self._components.values():
            if isinstance(c, (DCVoltageSource, ACVoltageSource)):
                battery_dsu.union(canonical(c.nodes[0]), canonical(c.nodes[1]))

        return battery_dsu.find(c_a) == battery_dsu.find(c_b)

    def validate_topology(
        self, *, regularize_ideal_loops: bool = False
    ) -> list[CircuitWarning]:
        """Validate the circuit graph for fatal errors and non-fatal warnings.

        Checks performed:
            1. At least one `Ground` component is connected (`NoGroundError`).
            2. Every node has at least 2 terminal connections and a path to
               ground (`FloatingNodeError`).
            3. Ideal voltage sources do not form a cycle or short to ground
               (`VoltageSourceLoopError`), unless `regularize_ideal_loops=True`,
               which records `ideal_loop_regularized` warnings instead.
            4. Ammeters are not connected directly in parallel across fixed
               voltage sources or shorted nodes (`CircuitWarning`).

        Returns:
            List of `CircuitWarning` objects discovered during inspection.
        """
        warnings: list[CircuitWarning] = []
        gnd_nodes = self.grounded_nodes()
        if not gnd_nodes:
            raise NoGroundError(
                "No ground connected — Add a Ground part and wire it to the circuit"
            )

        non_ground_comps = [
            c for c in self._components.values() if not isinstance(c, Ground)
        ]
        if not non_ground_comps:
            return warnings

        CANONICAL_GND = "__GND__"

        def canonical(node: str) -> str:
            return CANONICAL_GND if node in gnd_nodes else node

        terminal_degree: dict[str, int] = defaultdict(int)
        adjacency: dict[str, set[str]] = defaultdict(set)
        gates: list[LogicGate] = []

        for comp in non_ground_comps:
            if isinstance(comp, LogicGate):
                gates.append(comp)
                # Every gate terminal returns internally to implicit GND ("0"), so
                # unloaded outputs or high-Z inputs stay solvable (pin-level checks handle
                # undriven gate inputs explicitly via `floating_gate_input` warnings).
                for in_node in comp.input_nodes:
                    c_in = canonical(in_node)
                    terminal_degree[in_node] += 2
                    adjacency[c_in].add(CANONICAL_GND)
                    adjacency[CANONICAL_GND].add(c_in)
                c_out = canonical(comp.output_node)
                terminal_degree[comp.output_node] += 2
                adjacency[c_out].add(CANONICAL_GND)
                adjacency[CANONICAL_GND].add(c_out)
                continue
            if isinstance(comp, MCUBlock):
                for p_node in comp.nodes:
                    c_pin = canonical(p_node)
                    terminal_degree[p_node] += 2
                    adjacency[c_pin].add(CANONICAL_GND)
                    adjacency[CANONICAL_GND].add(c_pin)
                continue

            if isinstance(comp, Potentiometer):
                for u, v in ((comp.node_a, comp.node_w), (comp.node_w, comp.node_b)):
                    cu, cv = canonical(u), canonical(v)
                    terminal_degree[u] += 1
                    terminal_degree[v] += 1
                    adjacency[cu].add(cv)
                    adjacency[cv].add(cu)
                continue

            if isinstance(comp, BJTBase):
                cb, cc, ce = canonical(comp.node_b), canonical(comp.node_c), canonical(comp.node_e)
                terminal_degree[comp.node_b] += 1
                terminal_degree[comp.node_c] += 1
                terminal_degree[comp.node_e] += 1
                adjacency[cb].update((cc, ce))
                adjacency[cc].update((cb, ce))
                adjacency[ce].update((cb, cc))
                continue

            if len(comp.nodes) < 2:
                continue
            n_a, n_b = comp.nodes[0], comp.nodes[1]
            c_a, c_b = canonical(n_a), canonical(n_b)
            terminal_degree[n_a] += 1
            terminal_degree[n_b] += 1
            adjacency[c_a].add(c_b)
            adjacency[c_b].add(c_a)

        # 1. Check for dangling nodes (only 1 connection on a non-ground node).
        for node in sorted(self.all_nodes() - gnd_nodes):
            if terminal_degree[node] < 2:
                raise FloatingNodeError(node, f"Node {node} is floating")

        # 2. Check that every node has a topological path to ground.
        visited: set[str] = {CANONICAL_GND}
        queue: deque[str] = deque([CANONICAL_GND])
        while queue:
            curr = queue.popleft()
            for neighbor in adjacency.get(curr, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        for node in sorted(self.all_nodes() - gnd_nodes):
            if canonical(node) not in visited:
                raise FloatingNodeError(node, f"Node {node} is floating")

        # 3. Check for ideal voltage-source loops using Union-Find.
        vs_dsu = _DisjointSet()
        seen_ideal_vs: list[tuple[str, str, str]] = []
        for comp in non_ground_comps:
            is_ideal_vs = isinstance(comp, (DCVoltageSource, ACVoltageSource)) and (
                comp.r_internal == 0.0
            )
            if is_ideal_vs:
                c_a = canonical(comp.nodes[0])
                c_b = canonical(comp.nodes[1])
                if not vs_dsu.union(c_a, c_b):
                    loop_root = vs_dsu.find(c_a)
                    loop_sources = [
                        name
                        for name, na, nb in seen_ideal_vs
                        if vs_dsu.find(na) == loop_root
                    ] + [comp.name]
                    if regularize_ideal_loops:
                        warnings.append(
                            CircuitWarning(
                                code="ideal_loop_regularized",
                                message=(
                                    f"Ideal voltage source loop ({', '.join(loop_sources)}) "
                                    "regularized with small series impedance (1 uOhm)."
                                ),
                                component_names=tuple(loop_sources),
                            )
                        )
                    else:
                        raise VoltageSourceLoopError(
                            sources=tuple(loop_sources),
                            message="Voltage sources in a loop",
                        )
                seen_ideal_vs.append((comp.name, c_a, c_b))

        # 4. Check for ammeters connected directly across a rigid voltage difference.
        battery_dsu = _DisjointSet()
        for comp in non_ground_comps:
            if isinstance(comp, (DCVoltageSource, ACVoltageSource)):
                battery_dsu.union(canonical(comp.nodes[0]), canonical(comp.nodes[1]))

        for comp in non_ground_comps:
            if isinstance(comp, Ammeter):
                c_a = canonical(comp.nodes[0])
                c_b = canonical(comp.nodes[1])
                if c_a == c_b or battery_dsu.find(c_a) == battery_dsu.find(c_b):
                    warnings.append(
                        CircuitWarning(
                            code="ammeter_in_parallel",
                            message=(
                                f"Ammeter {comp.name} is connected in parallel across "
                                "a voltage source or short circuit."
                            ),
                            component_names=(comp.name,),
                        )
                    )

        # 5. Digital logic gate topology checks (undriven inputs, contention,
        #    shorted outputs, and zero-delay combinational feedback loops).
        if gates:
            self._validate_logic_gates(
                non_ground_comps=non_ground_comps,
                gates=gates,
                canonical=canonical,
                canonical_gnd=CANONICAL_GND,
                warnings=warnings,
            )

        return warnings

    def _validate_logic_gates(
        self,
        *,
        non_ground_comps: list[Component],
        gates: list[LogicGate],
        canonical: object,
        canonical_gnd: str,
        warnings: list[CircuitWarning],
    ) -> None:
        """Check logic gate input drivers, output contention/shorts, and zero-delay loops."""
        # Build external conductive graph (without gate r_in to GND) to detect floating inputs
        # and build passive signal-propagation graph (without GND/ideal sources) for cycle checks.
        external_adj: dict[str, set[str]] = defaultdict(set)
        passive_signal_adj: dict[str, set[str]] = defaultdict(set)
        driven_seeds: set[str] = {canonical_gnd}
        wire_short_dsu = _DisjointSet()

        for comp in non_ground_comps:
            if isinstance(comp, Potentiometer):
                for u, v in ((comp.node_a, comp.node_w), (comp.node_w, comp.node_b)):
                    cu, cv = canonical(u), canonical(v)
                    external_adj[cu].add(cv)
                    external_adj[cv].add(cu)
                    if cu != canonical_gnd and cv != canonical_gnd:
                        passive_signal_adj[cu].add(cv)
                        passive_signal_adj[cv].add(cu)
                continue

            if isinstance(comp, LogicGate):
                driven_seeds.add(canonical(comp.output_node))
                continue
            if isinstance(comp, MCUBlock):
                for pin, p_node in zip(comp.pin_names, comp.nodes):
                    if comp.is_output_pin(pin):
                        driven_seeds.add(canonical(p_node))
                continue
            if isinstance(comp, Potentiometer):
                for u, v in ((comp.node_a, comp.node_w), (comp.node_w, comp.node_b)):
                    cu, cv = canonical(u), canonical(v)
                    terminal_degree[u] += 1
                    terminal_degree[v] += 1
                    adjacency[cu].add(cv)
                    adjacency[cv].add(cu)
                continue

            if isinstance(comp, BJTBase):
                cb, cc, ce = canonical(comp.node_b), canonical(comp.node_c), canonical(comp.node_e)
                terminal_degree[comp.node_b] += 1
                terminal_degree[comp.node_c] += 1
                terminal_degree[comp.node_e] += 1
                adjacency[cb].update((cc, ce))
                adjacency[cc].update((cb, ce))
                adjacency[ce].update((cb, cc))
                continue

            if len(comp.nodes) < 2:
                continue
            c_a = canonical(comp.nodes[0])
            c_b = canonical(comp.nodes[1])

            if isinstance(comp, (DCVoltageSource, ACVoltageSource)):
                driven_seeds.add(c_a)
                driven_seeds.add(c_b)
                external_adj[c_a].add(c_b)
                external_adj[c_b].add(c_a)
                continue

            if isinstance(comp, DCCurrentSource):
                driven_seeds.add(c_a)
                driven_seeds.add(c_b)
                external_adj[c_a].add(c_b)
                external_adj[c_b].add(c_a)
                continue

            if isinstance(comp, Voltmeter):
                # 10 MΩ measurement probe is high-Z and not a valid logic driver path.
                continue

            if isinstance(comp, Switch) and not comp.closed:
                # An open switch acts as ~1 GΩ isolation, leaving un-pulled inputs floating.
                continue

            if isinstance(comp, Ammeter) and getattr(comp, "is_ideal", False):
                wire_short_dsu.union(c_a, c_b)

            external_adj[c_a].add(c_b)
            external_adj[c_b].add(c_a)
            if c_a != canonical_gnd and c_b != canonical_gnd:
                passive_signal_adj[c_a].add(c_b)
                passive_signal_adj[c_b].add(c_a)

        # Compute all nodes reachable from at least one external signal/power driver.
        driven_nodes: set[str] = set(driven_seeds)
        d_queue: deque[str] = deque(driven_seeds)
        while d_queue:
            curr = d_queue.popleft()
            for nb in external_adj.get(curr, ()):
                if nb not in driven_nodes:
                    driven_nodes.add(nb)
                    d_queue.append(nb)

        for gate in gates:
            for pin_name, in_node in zip(gate.input_pin_names, gate.input_nodes):
                if canonical(in_node) not in driven_nodes:
                    warnings.append(
                        CircuitWarning(
                            code="floating_gate_input",
                            message=(
                                f"Floating input on {gate.name}.{pin_name} (node {in_node}): "
                                "no driver or pull resistor connected; Tie input to High/Low or wire to a driver"
                            ),
                            component_names=(gate.name,),
                        )
                    )

        # Output contention check: >= 2 gates driving the same net (or tied via ideal ammeters).
        gnd_short_root = wire_short_dsu.find(canonical_gnd)
        drivers_by_net: dict[str, list[LogicGate]] = defaultdict(list)
        for gate in gates:
            out_c = canonical(gate.output_node)
            out_root = wire_short_dsu.find(out_c)
            if out_root != gnd_short_root:
                drivers_by_net[out_root].append(gate)

        for drivers in drivers_by_net.values():
            if len(drivers) >= 2:
                g_names = tuple(g.name for g in drivers)
                v_high = max(g.v_oh for g in drivers)
                v_low = min(g.v_ol for g in drivers)
                r_pair = drivers[0].r_out + drivers[1].r_out
                i_worst = (v_high - v_low) / r_pair if r_pair > 0.0 else 0.0
                warnings.append(
                    CircuitWarning(
                        code="output_contention",
                        message=(
                            f"Logic output contention on net {drivers[0].output_node}: "
                            f"gates {', '.join(g_names)} drive the same node "
                            f"(opposing-level current up to {i_worst * 1e3:.1f} mA)"
                        ),
                        component_names=g_names,
                    )
                )

        # Check outputs shorted directly (or via ideal ammeter) to ground or an ideal voltage source.
        ideal_supply_roots: dict[str, list[str]] = defaultdict(list)
        for comp in non_ground_comps:
            if isinstance(comp, (DCVoltageSource, ACVoltageSource)) and comp.r_internal == 0.0:
                for term in comp.nodes[:2]:
                    root = wire_short_dsu.find(canonical(term))
                    if root != gnd_short_root and comp.name not in ideal_supply_roots[root]:
                        ideal_supply_roots[root].append(comp.name)

        for gate in gates:
            out_root = wire_short_dsu.find(canonical(gate.output_node))
            if out_root == gnd_short_root:
                i_short = max(abs(gate.v_oh), abs(gate.v_ol)) / gate.r_out
                warnings.append(
                    CircuitWarning(
                        code="output_shorted_to_supply",
                        message=(
                            f"Gate {gate.name} output is shorted to Ground: "
                            f"clamped at 0 V (up to {i_short * 1e3:.1f} mA short current)"
                        ),
                        component_names=(gate.name,),
                    )
                )
                continue

            direct_srcs = ideal_supply_roots.get(out_root, [])
            if direct_srcs:
                src_list = ", ".join(direct_srcs)
                warnings.append(
                    CircuitWarning(
                        code="output_shorted_to_supply",
                        message=(
                            f"Gate {gate.name} output is shorted to voltage source ({src_list}): "
                            "output pin cannot change node potential; add series resistance or disconnect source"
                        ),
                        component_names=(gate.name, *direct_srcs),
                    )
                )

        # Build directed zero-delay gate dependency graph to reject combinational cycles with 0 delay.
        inputs_by_node: dict[str, list[LogicGate]] = defaultdict(list)
        for gate in gates:
            if gate.t_pd != 0.0:
                continue
            for in_node in gate.input_nodes:
                c_in = canonical(in_node)
                if c_in != canonical_gnd:
                    inputs_by_node[c_in].append(gate)

        zero_gates = [g for g in gates if g.t_pd == 0.0]
        if not zero_gates:
            return

        zero_gate_adj: dict[str, list[LogicGate]] = defaultdict(list)
        for src_gate in zero_gates:
            out_c = canonical(src_gate.output_node)
            if out_c == canonical_gnd:
                continue
            reached_nodes: set[str] = {out_c}
            p_queue: deque[str] = deque([out_c])
            while p_queue:
                curr = p_queue.popleft()
                for dst_gate in inputs_by_node.get(curr, ()):
                    if dst_gate not in zero_gate_adj[src_gate.name]:
                        zero_gate_adj[src_gate.name].append(dst_gate)
                for nb in passive_signal_adj.get(curr, ()):
                    if nb not in reached_nodes:
                        reached_nodes.add(nb)
                        p_queue.append(nb)

        # DFS cycle detection on zero-delay gate subgraph.
        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done
        stack: list[str] = []

        def find_cycle(g_name: str) -> tuple[str, ...] | None:
            state[g_name] = 1
            stack.append(g_name)
            for nxt in zero_gate_adj.get(g_name, ()):
                nxt_state = state.get(nxt.name, 0)
                if nxt_state == 1:
                    idx = stack.index(nxt.name)
                    return tuple(stack[idx:])
                if nxt_state == 0:
                    cyc = find_cycle(nxt.name)
                    if cyc is not None:
                        return cyc
            stack.pop()
            state[g_name] = 2
            return None

        for g in zero_gates:
            if state.get(g.name, 0) == 0:
                cycle = find_cycle(g.name)
                if cycle is not None:
                    raise ZeroDelayLoopError(
                        gates=cycle,
                        message=(
                            "Zero-delay combinational feedback loop detected across "
                            f"{' -> '.join(cycle)}: set non-zero propagation delay t_pd > 0"
                        ),
                    )

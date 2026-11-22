"""Qt-free truth-table exhaustively simulated across DC operating points."""

from __future__ import annotations

import copy
import csv
import io
import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from circuitsim.components.logic_gate import LogicGate
from circuitsim.components.logic_probe import LogicProbe
from circuitsim.components.meters import Voltmeter
from circuitsim.components.sources import DCVoltageSource
from circuitsim.components.switch import Switch
from circuitsim.components.led import LED
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.solver import MNASolver


@dataclass(frozen=True)
class TruthTableRow:
    """Single row of a simulated truth table."""

    inputs: tuple[int, ...]
    outputs: tuple[str, ...]
    output_voltages: tuple[float, ...]


@dataclass(frozen=True)
class TruthTableResult:
    """Complete truth table over `2^N` binary input combinations."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    rows: tuple[TruthTableRow, ...]

    def as_mapping(self) -> dict[tuple[int, ...], tuple[str, ...]]:
        return {row.inputs: row.outputs for row in self.rows}

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([*self.input_names, *self.output_names])
        for row in self.rows:
            writer.writerow([*row.inputs, *row.outputs])
        return buf.getvalue()

    def to_markdown(self) -> str:
        headers = [*self.input_names, *self.output_names]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in self.rows:
            vals = [*map(str, row.inputs), *row.outputs]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def _extract_circuit(circuit_or_scene: Any) -> Circuit:
    if isinstance(circuit_or_scene, Circuit):
        return circuit_or_scene
    if hasattr(circuit_or_scene, "extract_connectivity_result"):
        return circuit_or_scene.extract_connectivity_result().circuit
    if hasattr(circuit_or_scene, "build_circuit"):
        return circuit_or_scene.build_circuit()
    raise TypeError("Expected a Circuit or a CircuitScene with connectivity extraction.")


def detect_truth_table_candidates(
    circuit_or_scene: Any,
) -> tuple[list[str], list[str]]:
    """Auto-detect `(candidate_input_names, candidate_output_names)` for a circuit."""
    circuit = _extract_circuit(circuit_or_scene)
    comps = circuit.flat_components()

    gate_in_nodes: set[str] = set()
    gate_out_nodes: set[str] = set()
    for comp in comps:
        if isinstance(comp, LogicGate):
            gate_in_nodes.update(comp.input_nodes)
            gate_out_nodes.add(comp.output_node)

    all_inputs = [c for c in comps if isinstance(c, (Switch, DCVoltageSource))]
    direct_inputs = [
        c for c in all_inputs if any(n in gate_in_nodes for n in c.nodes)
    ]
    chosen_inputs = [c.name for c in (direct_inputs or all_inputs)]

    probes = [c.name for c in comps if isinstance(c, LogicProbe)]
    if probes:
        chosen_outputs = probes
    else:
        meters_leds = [c.name for c in comps if isinstance(c, (Voltmeter, LED))]
        if meters_leds:
            chosen_outputs = meters_leds
        else:
            leaf_gates = [
                c.name
                for c in comps
                if isinstance(c, LogicGate) and c.output_node not in gate_in_nodes
            ]
            chosen_outputs = leaf_gates or [
                c.name for c in comps if isinstance(c, LogicGate)
            ]

    return chosen_inputs, chosen_outputs


def generate_truth_table(
    circuit_or_scene: Any,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    *,
    v_high: float = 5.0,
    v_low: float = 0.0,
    v_il: float | None = None,
    v_ih: float | None = None,
) -> TruthTableResult:
    """Exhaustively evaluate DC logic outputs over all `2^N` binary input combinations (N <= 6)."""
    base_circuit = _extract_circuit(circuit_or_scene)
    auto_inputs, auto_outputs = detect_truth_table_candidates(base_circuit)

    inputs = tuple(input_names) if input_names is not None else tuple(auto_inputs)
    outputs = tuple(output_names) if output_names is not None else tuple(auto_outputs)

    if not inputs:
        raise ValueError("No binary input sources or switches found for truth-table generation.")
    if len(inputs) > 6:
        raise ValueError(
            f"Truth-table generator supports at most 6 inputs (got {len(inputs)})."
        )
    if not outputs:
        raise ValueError("No output probes, meters, LEDs, or logic gates found.")

    rows: list[TruthTableRow] = []
    for bits in itertools.product((0, 1), repeat=len(inputs)):
        cloned = copy.deepcopy(base_circuit)
        for iname, bit in zip(inputs, bits):
            comp = cloned.get_component(iname)
            if isinstance(comp, Switch):
                comp.set_closed(bool(bit))
            elif isinstance(comp, DCVoltageSource):
                comp.dc_voltage = float(v_high if bit else v_low)
            else:
                raise TypeError(
                    f"Truth-table input {iname!r} ({comp.component_type}) must be a DCVoltageSource or Switch."
                )

        res = MNASolver(cloned).solve_dc()
        out_states: list[str] = []
        out_volts: list[float] = []
        for out_key in outputs:
            try:
                out_comp = cloned.get_component(out_key)
            except KeyError:
                out_comp = None

            if isinstance(out_comp, LogicProbe):
                v_out = float(res.node_voltages.get(out_comp.nodes[0], 0.0))
                eff_vil = float(v_il) if v_il is not None else out_comp.v_il
                eff_vih = float(v_ih) if v_ih is not None else out_comp.v_ih
            elif isinstance(out_comp, LogicGate):
                v_out = float(res.node_voltages.get(out_comp.output_node, 0.0))
                eff_vil = 1.5 if v_il is None else float(v_il)
                eff_vih = 3.5 if v_ih is None else float(v_ih)
            elif out_comp is not None:
                v_out = float(res.node_voltages.get(out_comp.nodes[0], 0.0))
                eff_vil = 1.5 if v_il is None else float(v_il)
                eff_vih = 3.5 if v_ih is None else float(v_ih)
            else:
                v_out = float(res.node_voltages.get(out_key, 0.0))
                eff_vil = 1.5 if v_il is None else float(v_il)
                eff_vih = 3.5 if v_ih is None else float(v_ih)

            if v_out >= eff_vih:
                bit_str = "1"
            elif v_out <= eff_vil:
                bit_str = "0"
            else:
                bit_str = "?"
            out_states.append(bit_str)
            out_volts.append(v_out)

        rows.append(
            TruthTableRow(
                inputs=tuple(bits),
                outputs=tuple(out_states),
                output_voltages=tuple(out_volts),
            )
        )

    return TruthTableResult(
        input_names=inputs,
        output_names=outputs,
        rows=tuple(rows),
    )

"""Qt-free standard SPICE netlist (.cir / .sp) exporter for Ohmbench schematics."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
from typing import Any

from circuitsim.components import (
    ACVoltageSource,
    Ammeter,
    BJTBase,
    Capacitor,
    Comparator,
    DCCurrentSource,
    DCVoltageSource,
    DigitalSequentialBlock,
    Diode,
    Fuse,
    Ground,
    Inductor,
    LED,
    Lightbulb,
    LogicGate,
    MCUBlock,
    MOSFETBase,
    NE555,
    OpAmp,
    Oscilloscope,
    Potentiometer,
    Resistor,
    SPDTRelay,
    Switch,
    Transformer,
    Voltmeter,
    ZenerDiode,
)
from circuitsim.engine.netlist import Circuit

_SPICE_SCALE_TABLE: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "MEG"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "u"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)

_AUTO_NODE_RE = re.compile(r"^N\d+$", re.IGNORECASE)


def format_spice_value(value: float) -> str:
    """Format a scalar number using standard SPICE suffixes (`MEG` for mega, `m` for milli)."""
    val = float(value)
    if not math.isfinite(val):
        return "0"
    if abs(val) < 1e-18:
        return "0"
    sign = "-" if val < 0.0 else ""
    mag = abs(val)
    for threshold, suffix in _SPICE_SCALE_TABLE:
        if mag >= threshold * (1.0 - 1e-12):
            scaled = mag / threshold
            txt = f"{scaled:.6g}"
            return f"{sign}{txt}{suffix}"
    return f"{val:.6g}"


def _resolve_circuit_and_labels(scene_or_graph: Any) -> tuple[Circuit, set[str]]:
    """Duck-type accept `Circuit`, `CircuitScene`, `CircuitDocumentV2`, or `(WireGraph, placed)`."""
    explicit_labels: set[str] = set()

    if isinstance(scene_or_graph, Circuit):
        for comp in scene_or_graph.flat_components():
            for n in comp.nodes:
                if n != "0" and not _AUTO_NODE_RE.match(n):
                    explicit_labels.add(n)
        return scene_or_graph, explicit_labels

    # CircuitScene (duck-typed cleanly without importing PySide6)
    if hasattr(scene_or_graph, "extract_connectivity_result"):
        conn = scene_or_graph.extract_connectivity_result()
        wg = getattr(scene_or_graph, "wire_graph", None)
        if wg is not None:
            for w in getattr(wg, "wires", ()):
                lbl = str(getattr(w, "net_label", "") or "").strip()
                if lbl:
                    explicit_labels.add(lbl)
            for j in getattr(wg, "junctions", ()):
                lbl = str(getattr(j, "net_label", "") or "").strip()
                if lbl:
                    explicit_labels.add(lbl)
        return conn.circuit, explicit_labels

    # CircuitDocumentV2 or (WireGraph, placed)
    from circuitsim.wiring.netlist import build_netlist_from_wire_graph

    if hasattr(scene_or_graph, "graph") and hasattr(scene_or_graph, "placed"):
        graph = scene_or_graph.graph
        placed = scene_or_graph.placed
    elif isinstance(scene_or_graph, (tuple, list)) and len(scene_or_graph) == 2:
        graph, placed = scene_or_graph
    else:
        raise TypeError(
            f"Expected Circuit, CircuitScene, CircuitDocumentV2, or (WireGraph, placed), got {type(scene_or_graph)!r}"
        )

    for w in getattr(graph, "wires", ()):
        lbl = str(getattr(w, "net_label", "") or "").strip()
        if lbl:
            explicit_labels.add(lbl)
    for j in getattr(graph, "junctions", ()):
        lbl = str(getattr(j, "net_label", "") or "").strip()
        if lbl:
            explicit_labels.add(lbl)
    specs = [getattr(p, "component", p) for p in placed]
    nl = build_netlist_from_wire_graph(graph, specs)
    return nl.circuit, explicit_labels


def _spice_card_name(prefix: str, comp_name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", comp_name)
    if clean.upper().startswith(prefix.upper()):
        return clean
    return f"{prefix}{clean}"


def export_to_spice_netlist(
    scene_or_graph: Any,
    title: str = "Ohmbench Circuit",
    *,
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """Export an Ohmbench circuit/schematic into a standard SPICE `.cir` netlist string."""
    circuit, explicit_labels = _resolve_circuit_and_labels(scene_or_graph)

    # Map nodes: Ground -> "0", preserve named labels, unnamed -> N001, N002, ...
    grounded = circuit.grounded_nodes()
    node_map: dict[str, str] = {g: "0" for g in grounded}
    used_names: set[str] = {"0"}
    unnamed_order: list[str] = []

    for comp in circuit.flat_components():
        for raw_n in comp.nodes:
            if raw_n in node_map:
                continue
            if raw_n in explicit_labels or (not _AUTO_NODE_RE.match(raw_n) and raw_n != "0"):
                # Keep user-given net labels (case-preserved)
                if raw_n.upper() == "GND":
                    node_map[raw_n] = "0"
                else:
                    node_map[raw_n] = raw_n
                    used_names.add(raw_n)
            elif raw_n not in unnamed_order:
                unnamed_order.append(raw_n)

    next_num = 1
    for raw_n in unnamed_order:
        while f"N{next_num:03d}" in used_names:
            next_num += 1
        mapped = f"N{next_num:03d}"
        next_num += 1
        used_names.add(mapped)
        node_map[raw_n] = mapped

    def n_str(raw_n: str) -> str:
        return node_map.get(raw_n, "0" if raw_n in grounded else raw_n)

    lines: list[str] = [f"* {title}", "* Exported from Ohmbench"]
    model_cards: list[str] = []
    seen_models: set[str] = set()

    for comp in circuit.flat_components():
        if isinstance(comp, Ground):
            continue

        if isinstance(comp, Resistor):
            cname = _spice_card_name("R", comp.name)
            lines.append(f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(comp.resistance)}")
        elif isinstance(comp, Potentiometer):
            r_tot = comp.resistance
            r_aw = max(1e-3, r_tot * comp.alpha)
            r_wb = max(1e-3, r_tot * (1.0 - comp.alpha))
            base = re.sub(r"[^A-Za-z0-9_]", "_", comp.name)
            lines.append(f"R{base}_AW {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(r_aw)}")
            lines.append(f"R{base}_WB {n_str(comp.nodes[1])} {n_str(comp.nodes[2])} {format_spice_value(r_wb)}")
        elif isinstance(comp, Capacitor):
            cname = _spice_card_name("C", comp.name)
            ic_txt = f" IC={format_spice_value(comp.initial_voltage)}" if abs(comp.initial_voltage) > 0.0 else ""
            lines.append(f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(comp.capacitance)}{ic_txt}")
        elif isinstance(comp, Inductor):
            cname = _spice_card_name("L", comp.name)
            ic_txt = f" IC={format_spice_value(comp.initial_current)}" if abs(comp.initial_current) > 0.0 else ""
            lines.append(f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(comp.inductance)}{ic_txt}")
        elif isinstance(comp, DCVoltageSource):
            cname = _spice_card_name("V", comp.name)
            lines.append(f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} DC {format_spice_value(comp.dc_voltage)}")
        elif isinstance(comp, ACVoltageSource):
            cname = _spice_card_name("V", comp.name)
            lines.append(
                f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} "
                f"SIN({format_spice_value(comp.dc_offset)} {format_spice_value(comp.amplitude)} {format_spice_value(comp.frequency)})"
            )
        elif isinstance(comp, DCCurrentSource):
            cname = _spice_card_name("I", comp.name)
            lines.append(f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} DC {format_spice_value(comp.dc_current)}")
        elif isinstance(comp, Switch):
            r_eq = comp.r_on if comp.closed else comp.r_off
            cname = _spice_card_name("R", f"SW_{comp.name}")
            lines.append(
                f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(r_eq)} ; Switch {comp.name} ({'ON' if comp.closed else 'OFF'})"
            )
        elif isinstance(comp, Fuse):
            r_eq = comp.r_blown_ohm if comp.is_blown else comp.r_on_ohm
            cname = _spice_card_name("R", f"F_{comp.name}")
            lines.append(
                f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(r_eq)} ; Fuse {comp.name} rating={format_spice_value(comp.rating_amps)}A"
            )
        elif isinstance(comp, Lightbulb):
            cname = _spice_card_name("R", f"LAMP_{comp.name}")
            lines.append(
                f"{cname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(comp.r_hot)} ; Lightbulb {comp.name} ({comp.rated_power:g}W/{comp.rated_voltage:g}V)"
            )
        elif isinstance(comp, ZenerDiode):
            dname = _spice_card_name("D", comp.name)
            mname = f"DZ_{re.sub(r'[^A-Za-z0-9_]', '_', comp.name)}"
            lines.append(f"{dname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {mname}")
            if mname not in seen_models:
                seen_models.add(mname)
                model_cards.append(f".MODEL {mname} D(IS={comp.is_sat:.4g} N={comp.n:.4g} RS={comp.rs:.4g} BV={comp.v_z:.4g})")
        elif isinstance(comp, LED):
            dname = _spice_card_name("D", comp.name)
            mname = f"DLED_{re.sub(r'[^A-Za-z0-9_]', '_', comp.name)}"
            lines.append(f"{dname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {mname} ; LED {comp.color_name}")
            if mname not in seen_models:
                seen_models.add(mname)
                model_cards.append(f".MODEL {mname} D(IS=1e-12 N=2.0)")
        elif isinstance(comp, Diode):
            dname = _spice_card_name("D", comp.name)
            raw_m = comp.model_name.strip() or f"D_{comp.name}"
            mname = f"DMOD_{re.sub(r'[^A-Za-z0-9_]', '_', raw_m)}"
            lines.append(f"{dname} {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {mname}")
            if mname not in seen_models:
                seen_models.add(mname)
                model_cards.append(f".MODEL {mname} D(IS={comp.is_sat:.4g} N={comp.n:.4g} RS={comp.rs:.4g} BV={comp.bv:.4g})")
        elif isinstance(comp, BJTBase):
            # BJTBase.nodes = (B, C, E); standard SPICE Q is (C, B, E)
            qname = _spice_card_name("Q", comp.name)
            kind = "NPN" if comp.component_type == "npn_bjt" else "PNP"
            mname = f"{kind}_{re.sub(r'[^A-Za-z0-9_]', '_', comp.name)}"
            lines.append(f"{qname} {n_str(comp.nodes[1])} {n_str(comp.nodes[0])} {n_str(comp.nodes[2])} {mname}")
            if mname not in seen_models:
                seen_models.add(mname)
                model_cards.append(f".MODEL {mname} {kind}(BF={comp.beta_f:.4g} BR={comp.beta_r:.4g} IS={comp.i_s:.4g} VAF={comp.v_af:.4g})")
        elif isinstance(comp, MOSFETBase):
            # MOSFETBase.nodes = (G, D, S); standard SPICE M is (D, G, S, B)
            mcard = _spice_card_name("M", comp.name)
            kind = "NMOS" if comp.component_type == "nmos" else "PMOS"
            mname = f"{kind}_{re.sub(r'[^A-Za-z0-9_]', '_', comp.name)}"
            ns = n_str(comp.nodes[2])
            lines.append(f"{mcard} {n_str(comp.nodes[1])} {n_str(comp.nodes[0])} {ns} {ns} {mname}")
            if mname not in seen_models:
                seen_models.add(mname)
                model_cards.append(f".MODEL {mname} {kind}(VTO={comp.v_th:.4g} KP={comp.k_p:.4g} LAMBDA={comp.lambda_:.4g})")
        elif isinstance(comp, Transformer):
            base = re.sub(r"[^A-Za-z0-9_]", "_", comp.name)
            lp = comp.l_primary_h
            ls = lp * ((comp.secondary_turns / max(comp.primary_turns, 1.0)) ** 2)
            lines.append(f"L{base}_P {n_str(comp.nodes[0])} {n_str(comp.nodes[1])} {format_spice_value(lp)}")
            lines.append(f"L{base}_S {n_str(comp.nodes[2])} {n_str(comp.nodes[3])} {format_spice_value(ls)}")
            lines.append(f"K{base} L{base}_P L{base}_S {comp.coupling_k:.6g}")
        elif isinstance(comp, Oscilloscope):
            lines.append(
                f"* Oscilloscope {comp.name} CH1={n_str(comp.nodes[0])} CH2={n_str(comp.nodes[1])} "
                f"EXT_TRIG={n_str(comp.nodes[2])} GND={n_str(comp.nodes[3])} RIN={format_spice_value(comp.r_in)}"
            )
        elif isinstance(comp, (Voltmeter, Ammeter)):
            lines.append(
                f"* {comp.__class__.__name__} {comp.name} ({n_str(comp.nodes[0])}, {n_str(comp.nodes[1])}) R={format_spice_value(comp.r_internal)}"
            )
        elif isinstance(comp, (OpAmp, Comparator, NE555, MCUBlock, DigitalSequentialBlock, LogicGate, SPDTRelay)):
            pin_list = " ".join(n_str(n) for n in comp.nodes)
            lines.append(f"* {comp.__class__.__name__} {comp.name} nodes=[{pin_list}]")
        else:
            pin_list = " ".join(n_str(n) for n in comp.nodes)
            lines.append(f"* {comp.component_type} {comp.name} nodes=[{pin_list}]")

    if model_cards:
        lines.append("*")
        lines.extend(model_cards)
    lines.append(".end")
    netlist_text = "\n".join(lines) + "\n"

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(netlist_text, encoding="utf-8")

    return netlist_text

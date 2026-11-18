"""Qt-free 3rd-party SPICE (.MODEL / .SUBCKT) importer with pin mapper and encrypted detector."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from circuitsim.components import (
    Capacitor,
    Component,
    DCCurrentSource,
    DCVoltageSource,
    Diode,
    Inductor,
    NPNBJT,
    PNPBJT,
    Resistor,
    ZenerDiode,
)


class SpiceImportError(ValueError):
    """Raised when a SPICE model, subcircuit, or pin mapping is invalid."""


class EncryptedModelError(SpiceImportError):
    """Raised when vendor-encrypted SPICE content (e.g. PSpice / HSPICE) is detected."""


_SPICE_SCALE_SUFFIXES: dict[str, float] = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "mil": 25.4e-6,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
    "t": 1e12,
}

_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ]*)\s*$"
)


def parse_spice_number(raw: str | float | int) -> float:
    """Parse a SPICE numeric literal with classic suffix rules (`m`=1e-3, `meg`=1e6)."""
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        raise SpiceImportError(f"Invalid SPICE numeric value {raw!r}.")
    mantissa = float(match.group(1))
    suffix_raw = match.group(2).lower()
    if not suffix_raw:
        return mantissa
    for prefix in ("meg", "mil", "f", "p", "n", "u", "µ", "m", "k", "g", "t"):
        if suffix_raw.startswith(prefix):
            return mantissa * _SPICE_SCALE_SUFFIXES[prefix]
    # Pure unit suffix (e.g. 'v', 'a', 'ohm', 'hz', 's') with multiplier 1
    if suffix_raw in ("v", "a", "ohm", "ohms", "hz", "s", "h"):
        return mantissa
    raise SpiceImportError(f"Unrecognized SPICE unit/suffix {suffix_raw!r} in {raw!r}.")


_ENCRYPTED_MARKERS: tuple[tuple[str, str], ...] = (
    ("$cdnencstart", "Cadence/PSpice encrypted block ($CDNENCSTART)"),
    (".prot", "HSPICE encrypted block (.prot / .protect)"),
    ("`pragma protect", "IEEE 1735 encrypted SPICE pragma (`pragma protect)"),
    ("*encrypted", "Encrypted SPICE model header (*ENCRYPTED)"),
    (".encrypted", "Encrypted SPICE directive (.ENCRYPTED)"),
)


def detect_encrypted_spice(raw: str | bytes) -> str | None:
    """Detect PSpice/HSPICE/IEEE-1735 encryption markers or binary ciphertext."""
    if isinstance(raw, bytes):
        if b"\x00" in raw[:4096]:
            return "Binary / NUL-byte encrypted payload detected"
        sample = raw[:16384]
        non_text = sum(1 for b in sample if b < 9 or (14 <= b <= 31))
        if len(sample) > 0 and (non_text / len(sample)) > 0.05:
            return "Binary non-ASCII encrypted SPICE payload detected"
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = raw
        if "\x00" in text:
            return "Binary / NUL-byte encrypted payload detected"

    lowered = text.lower()
    for marker, reason in _ENCRYPTED_MARKERS:
        if marker in lowered:
            return reason
    return None


@dataclass(frozen=True)
class SpiceModelDef:
    """Parsed SPICE `.MODEL <name> <type>(<params>)` definition."""

    name: str
    model_type: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SpiceElement:
    """Single element line inside a SPICE `.SUBCKT` or netlist."""

    refdes: str
    kind: str
    nodes: tuple[str, ...]
    value: float | None = None
    model_name: str | None = None
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SpiceSubcktDef:
    """Parsed SPICE `.SUBCKT <name> <pins...>` block."""

    name: str
    pins: tuple[str, ...]
    elements: tuple[SpiceElement, ...] = ()
    models: dict[str, SpiceModelDef] = field(default_factory=dict)


@dataclass(frozen=True)
class SpiceLibrary:
    """Container returned by `parse_spice_library` / `parse_spice_file`."""

    models: dict[str, SpiceModelDef] = field(default_factory=dict)
    subckts: dict[str, SpiceSubcktDef] = field(default_factory=dict)
    top_elements: tuple[SpiceElement, ...] = ()


def _unfold_spice_lines(text: str) -> list[str]:
    logical: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        # Strip inline SPICE comment markers `;` or ` $ `
        for c_sep in (";", " $ "):
            if c_sep in stripped:
                stripped = stripped.split(c_sep, 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith("+"):
            cont = stripped[1:].strip()
            if logical:
                logical[-1] = f"{logical[-1]} {cont}"
            else:
                logical.append(cont)
        else:
            logical.append(stripped)
    return logical


def _parse_kv_params(param_text: str) -> dict[str, float]:
    cleaned = param_text.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace(",", " ")
    params: dict[str, float] = {}
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s()]+)", cleaned):
        k = m.group(1).upper()
        try:
            params[k] = parse_spice_number(m.group(2))
        except SpiceImportError:
            continue
    return params


def _parse_model_line(line: str) -> SpiceModelDef:
    m = re.match(
        r"^\.MODEL\s+(\S+)\s+([A-Za-z0-9_]+)\s*(\(.*)?$",
        line,
        flags=re.IGNORECASE,
    )
    if not m:
        raise SpiceImportError(f"Malformed .MODEL directive: {line!r}")
    name = m.group(1)
    mtype_raw = m.group(2).upper()
    rest = m.group(3) or ""
    # Handle `.MODEL D1N4148 D IS=2.52n N=1.752` without outer parentheses too
    if not rest:
        tokens = line.split(None, 3)
        rest = tokens[3] if len(tokens) > 3 else ""
    params = _parse_kv_params(rest)
    return SpiceModelDef(name=name, model_type=mtype_raw, params=params)


def _parse_element_line(line: str) -> SpiceElement:
    tokens = line.split()
    if not tokens:
        raise SpiceImportError("Empty SPICE element line.")
    refdes = tokens[0]
    kind = refdes[0].upper()
    if kind in ("R", "C", "L"):
        if len(tokens) < 4:
            raise SpiceImportError(f"Expected 2 nodes and value on {kind} line: {line!r}")
        nodes = (tokens[1], tokens[2])
        val = parse_spice_number(tokens[3])
        return SpiceElement(refdes=refdes, kind=kind, nodes=nodes, value=val)
    if kind == "D":
        if len(tokens) < 4:
            raise SpiceImportError(f"Expected anode, cathode, and model on D line: {line!r}")
        nodes = (tokens[1], tokens[2])
        model_name = tokens[3]
        return SpiceElement(refdes=refdes, kind="D", nodes=nodes, model_name=model_name)
    if kind == "Q":
        if len(tokens) < 5:
            raise SpiceImportError(f"Expected C, B, E nodes and model on Q line: {line!r}")
        nodes = (tokens[1], tokens[2], tokens[3])
        model_name = tokens[4]
        return SpiceElement(refdes=refdes, kind="Q", nodes=nodes, model_name=model_name)
    if kind in ("V", "I"):
        if len(tokens) < 4:
            raise SpiceImportError(f"Expected 2 nodes and value on {kind} line: {line!r}")
        nodes = (tokens[1], tokens[2])
        # Support optional `DC 5.0` keyword
        val_tok = tokens[4] if (tokens[3].upper() == "DC" and len(tokens) >= 5) else tokens[3]
        val = parse_spice_number(val_tok)
        return SpiceElement(refdes=refdes, kind=kind, nodes=nodes, value=val)
    if kind == "X":
        if len(tokens) < 3:
            raise SpiceImportError(f"Malformed X subckt call: {line!r}")
        nodes = tuple(tokens[1:-1])
        model_name = tokens[-1]
        return SpiceElement(refdes=refdes, kind="X", nodes=nodes, model_name=model_name)
    raise SpiceImportError(f"Unsupported SPICE element {refdes!r} in {line!r}.")


def parse_spice_library(raw: str | bytes) -> SpiceLibrary:
    """Parse 3rd-party SPICE `.MODEL` and `.SUBCKT` definitions cleanly."""
    enc_reason = detect_encrypted_spice(raw)
    if enc_reason is not None:
        raise EncryptedModelError(
            f"Cannot import encrypted vendor SPICE model: {enc_reason}."
        )
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    lines = _unfold_spice_lines(text)

    top_models: dict[str, SpiceModelDef] = {}
    subckts: dict[str, SpiceSubcktDef] = {}
    top_elements: list[SpiceElement] = []

    current_subckt_name: str | None = None
    current_pins: tuple[str, ...] = ()
    current_elems: list[SpiceElement] = []
    current_models: dict[str, SpiceModelDef] = {}

    for line in lines:
        upper = line.upper()
        if upper.startswith(".SUBCKT "):
            if current_subckt_name is not None:
                raise SpiceImportError(f"Nested .SUBCKT is not supported ({line!r}).")
            parts = line.split()
            if len(parts) < 3:
                raise SpiceImportError(f".SUBCKT requires name and pins: {line!r}")
            current_subckt_name = parts[1]
            current_pins = tuple(parts[2:])
            current_elems = []
            current_models = {}
        elif upper.startswith(".ENDS"):
            if current_subckt_name is None:
                continue
            subckts[current_subckt_name] = SpiceSubcktDef(
                name=current_subckt_name,
                pins=current_pins,
                elements=tuple(current_elems),
                models=dict(current_models),
            )
            current_subckt_name = None
        elif upper.startswith(".MODEL "):
            mdef = _parse_model_line(line)
            if current_subckt_name is not None:
                current_models[mdef.name] = mdef
            else:
                top_models[mdef.name] = mdef
        elif upper.startswith("."):
            # Skip harmless SPICE control cards (.OPTIONS, .TEMP, .PARAM, .END)
            continue
        else:
            elem = _parse_element_line(line)
            if current_subckt_name is not None:
                current_elems.append(elem)
            else:
                top_elements.append(elem)

    if current_subckt_name is not None:
        raise SpiceImportError(f".SUBCKT {current_subckt_name!r} missing terminating .ENDS.")

    return SpiceLibrary(
        models=top_models,
        subckts=subckts,
        top_elements=tuple(top_elements),
    )


def parse_spice_file(path: str | os.PathLike[str]) -> SpiceLibrary:
    """Read and parse a `.lib` / `.mod` / `.cir` SPICE file, rejecting encrypted files."""
    raw = Path(path).read_bytes()
    return parse_spice_library(raw)


def instantiate_diode_model(
    model: SpiceModelDef,
    name: str,
    node_anode: str,
    node_cathode: str,
    *,
    as_zener: bool = False,
) -> Diode:
    """Construct a calibrated Shockley `Diode` or `ZenerDiode` from a SPICE `.MODEL`."""
    if not model.model_type.upper().startswith("D"):
        raise SpiceImportError(
            f"Model {model.name!r} has type {model.model_type!r}; expected Diode ('D')."
        )
    p = model.params
    if as_zener:
        return ZenerDiode(
            name=name,
            node_anode=node_anode,
            node_cathode=node_cathode,
            v_z=abs(p.get("BV", ZenerDiode.DEFAULT_VZ)),
            i_s=p.get("IS", ZenerDiode.DEFAULT_ZENER_IS),
            n=p.get("N", ZenerDiode.DEFAULT_ZENER_N),
            r_s=p.get("RS", ZenerDiode.DEFAULT_ZENER_RS),
            ibv=p.get("IBV", ZenerDiode.DEFAULT_ZENER_IBV),
            model_name=model.name,
        )
    return Diode(
        name=name,
        node_anode=node_anode,
        node_cathode=node_cathode,
        is_sat=p.get("IS", Diode.DEFAULT_IS),
        n=p.get("N", Diode.DEFAULT_N),
        rs=p.get("RS", Diode.DEFAULT_RS),
        bv=abs(p.get("BV", Diode.DEFAULT_BV)),
        ibv=p.get("IBV", Diode.DEFAULT_IBV),
        model_name=model.name,
    )


def instantiate_bjt_model(
    model: SpiceModelDef,
    name: str,
    node_b: str,
    node_c: str,
    node_e: str,
) -> NPNBJT | PNPBJT:
    """Construct a calibrated `NPNBJT` or `PNPBJT` from a SPICE `.MODEL ... NPN/PNP(...)`."""
    mtype = model.model_type.upper()
    if mtype not in ("NPN", "PNP"):
        raise SpiceImportError(
            f"Model {model.name!r} has type {model.model_type!r}; expected 'NPN' or 'PNP'."
        )
    p = model.params
    cls = NPNBJT if mtype == "NPN" else PNPBJT
    return cls(
        name=name,
        node_b=node_b,
        node_c=node_c,
        node_e=node_e,
        beta_f=p.get("BF", cls.DEFAULT_BF),
        beta_r=p.get("BR", cls.DEFAULT_BR),
        i_s=p.get("IS", cls.DEFAULT_IS),
        v_af=p.get("VAF", cls.DEFAULT_VAF),
        model_name=model.name,
    )


def map_subckt_pins(
    subckt: SpiceSubcktDef,
    pin_mapping: Mapping[str, str] | Sequence[str],
) -> dict[str, str]:
    """Validate and resolve SPICE `.SUBCKT` pin names to canonical terminal names.

    Args:
        subckt: Parsed `.SUBCKT` definition.
        pin_mapping: Either a mapping `{spice_pin: canonical_terminal}` or an ordered
            sequence `[canonical_for_pin_0, canonical_for_pin_1, ...]` matching `subckt.pins`.
    """
    n_pins = len(subckt.pins)
    if isinstance(pin_mapping, Mapping):
        missing = [p for p in subckt.pins if p not in pin_mapping]
        if missing:
            raise SpiceImportError(
                f"Subcircuit {subckt.name!r} missing pin mapping for: {', '.join(missing)}."
            )
        resolved = {p: str(pin_mapping[p]) for p in subckt.pins}
    else:
        seq = [str(x) for x in pin_mapping]
        if len(seq) != n_pins:
            raise SpiceImportError(
                f"Subcircuit {subckt.name!r} has {n_pins} pin(s) ({subckt.pins}), "
                f"got {len(seq)} canonical terminal names."
            )
        resolved = {p: c for p, c in zip(subckt.pins, seq)}

    canonical_vals = list(resolved.values())
    if len(set(canonical_vals)) != len(canonical_vals):
        raise SpiceImportError(
            f"Duplicate canonical terminal target in pin mapping for {subckt.name!r}: {resolved}"
        )
    return resolved


def instantiate_subckt_elements(
    subckt: SpiceSubcktDef,
    instance_name: str,
    pin_mapping: Mapping[str, str] | Sequence[str],
    terminal_to_node: Mapping[str, str],
    *,
    external_models: Mapping[str, SpiceModelDef] | None = None,
) -> list[Component]:
    """Expand a parsed `.SUBCKT` into solver components with explicit pin remapping."""
    pin_to_canon = map_subckt_pins(subckt, pin_mapping)
    model_lookup: dict[str, SpiceModelDef] = {}
    if external_models:
        model_lookup.update(external_models)
    model_lookup.update(subckt.models)

    def resolve_node(spice_node: str) -> str:
        if spice_node in ("0", "gnd", "GND"):
            return "0"
        if spice_node in pin_to_canon:
            canon = pin_to_canon[spice_node]
            if canon not in terminal_to_node:
                raise SpiceImportError(
                    f"Canonical terminal {canon!r} missing from terminal_to_node."
                )
            return terminal_to_node[canon]
        return f"{instance_name}__{spice_node}"

    components: list[Component] = []
    for elem in subckt.elements:
        cname = f"{instance_name}_{elem.refdes}"
        mapped_nodes = [resolve_node(n) for n in elem.nodes]
        if elem.kind == "R":
            assert elem.value is not None
            components.append(Resistor(cname, mapped_nodes[0], mapped_nodes[1], elem.value))
        elif elem.kind == "C":
            assert elem.value is not None
            components.append(Capacitor(cname, mapped_nodes[0], mapped_nodes[1], elem.value))
        elif elem.kind == "L":
            assert elem.value is not None
            components.append(Inductor(cname, mapped_nodes[0], mapped_nodes[1], elem.value))
        elif elem.kind == "V":
            assert elem.value is not None
            components.append(DCVoltageSource(cname, mapped_nodes[0], mapped_nodes[1], elem.value))
        elif elem.kind == "I":
            assert elem.value is not None
            components.append(DCCurrentSource(cname, mapped_nodes[0], mapped_nodes[1], elem.value))
        elif elem.kind == "D":
            mdef = model_lookup.get(elem.model_name or "")
            if mdef is None:
                components.append(
                    Diode(cname, mapped_nodes[0], mapped_nodes[1], model_name=elem.model_name or "1N4148")
                )
            else:
                components.append(
                    instantiate_diode_model(mdef, cname, mapped_nodes[0], mapped_nodes[1])
                )
        elif elem.kind == "Q":
            # Standard SPICE Q line order: Q<name> <collector> <base> <emitter> <model>
            mdef = model_lookup.get(elem.model_name or "")
            if mdef is None:
                components.append(NPNBJT(cname, mapped_nodes[1], mapped_nodes[0], mapped_nodes[2]))
            else:
                components.append(
                    instantiate_bjt_model(mdef, cname, mapped_nodes[1], mapped_nodes[0], mapped_nodes[2])
                )
        else:
            raise SpiceImportError(
                f"Subcircuit {subckt.name!r} element {elem.refdes!r} ({elem.kind}) not expandable."
            )
    return components

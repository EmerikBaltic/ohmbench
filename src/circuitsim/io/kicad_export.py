"""Qt-free KiCad 7/8 schematic (`.kicad_sch`) S-expression exporter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

from circuitsim.components.sources import ACVoltageSource, DCCurrentSource, DCVoltageSource
from circuitsim.io.circ_serializer import CircuitDocument, PlacedComponentSpec
from circuitsim.wiring.model import FreeEnd, JunctionRef, PointPx, TerminalRef

if TYPE_CHECKING:
    from circuitsim.wiring.serializer import CircuitDocumentV2

# 20 px in Ohmbench equals 2.54 mm (100 mil) standard KiCad schematic grid.
_PX_TO_MM: float = 2.54 / 20.0

_LIB_MAP: dict[str, tuple[str, str]] = {
    "resistor": ("Device:R", "R"),
    "capacitor": ("Device:C", "C"),
    "inductor": ("Device:L", "L"),
    "diode": ("Device:D", "D"),
    "zener_diode": ("Device:D_Zener", "DZ"),
    "led": ("Device:LED", "D"),
    "npn_bjt": ("Device:Q_NPN_BCE", "Q"),
    "pnp_bjt": ("Device:Q_PNP_BCE", "Q"),
    "nmos": ("Device:Q_NMOS_GDS", "M"),
    "pmos": ("Device:Q_PMOS_GDS", "M"),
    "opamp": ("Amplifier_Operational:LM358", "U"),
    "comparator": ("Comparator:LM393", "U"),
    "ne555": ("Timer:NE555P", "U"),
    "potentiometer": ("Device:R_Potentiometer", "RV"),
    "lightbulb": ("Device:Lamp", "LA"),
    "switch": ("Switch:SW_SPST", "SW"),
    "dc_voltage": ("Simulation_SPICE:VDC", "V"),
    "ac_voltage": ("Simulation_SPICE:VAC", "V"),
    "wav_voltage": ("Simulation_SPICE:VSOURCE", "V"),
    "dc_current": ("Simulation_SPICE:IDC", "I"),
    "voltmeter": ("Device:Voltmeter", "VM"),
    "ammeter": ("Device:Ammeter", "AM"),
    "ground": ("power:GND", "#PWR"),
}


def _esc(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _mm(px: float) -> float:
    return round(float(px) * _PX_TO_MM, 3)


def _deterministic_uuid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ohmbench-kicad:{seed}"))


def _display_value(spec: PlacedComponentSpec) -> str:
    comp = spec.component
    if hasattr(comp, "display_value"):
        try:
            return str(comp.display_value)
        except Exception:
            pass
    if isinstance(comp, DCVoltageSource):
        return f"{comp.dc_voltage:g}V"
    if isinstance(comp, ACVoltageSource):
        return f"{comp.amplitude:g}V {comp.frequency:g}Hz"
    if isinstance(comp, DCCurrentSource):
        return f"{comp.dc_current:g}A"
    return comp.component_type


def _lib_symbol_sexpr(lib_id: str) -> str:
    ref_prefix = next((rp for lid, rp in _LIB_MAP.values() if lid == lib_id), "U")
    is_gnd = lib_id == "power:GND"
    power_flag = "\n      (power)" if is_gnd else ""
    pin1_name = "GND" if is_gnd else "1"
    pin2 = (
        ""
        if is_gnd
        else '\n        (pin passive line (at 5.08 0 180) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))'
    )
    x1 = 0.0 if is_gnd else -5.08
    y1 = 0.0 if is_gnd else 0.0
    ang1 = 90 if is_gnd else 0
    return (
        f'    (symbol "{_esc(lib_id)}"{power_flag} (pin_numbers hide) (pin_names (offset 0))\n'
        f'      (in_bom {"no" if is_gnd else "yes"}) (on_board {"no" if is_gnd else "yes"})\n'
        f'      (property "Reference" "{_esc(ref_prefix)}" (at 0 2.54 0) (effects (font (size 1.27 1.27))))\n'
        f'      (property "Value" "{_esc(lib_id.split(":")[-1])}" (at 0 -2.54 0) (effects (font (size 1.27 1.27))))\n'
        f'      (symbol "{_esc(lib_id.split(":")[-1])}_0_1"\n'
        f'        (rectangle (start -2.54 -1.27) (end 2.54 1.27) (stroke (width 0.254) (type default)) (fill (type none)))\n'
        f'      )\n'
        f'      (symbol "{_esc(lib_id.split(":")[-1])}_1_1"\n'
        f'        (pin {"power_in" if is_gnd else "passive"} line (at {x1} {y1} {ang1}) (length 2.54) (name "{pin1_name}" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27))))){pin2}\n'
        f'      )\n'
        f'    )'
    )


def _attachment_point_px(
    att: TerminalRef | JunctionRef | FreeEnd,
    comp_map: dict[str, PlacedComponentSpec],
    junc_map: dict[str, PointPx],
) -> PointPx:
    if isinstance(att, FreeEnd):
        return (att.x, att.y)
    if isinstance(att, JunctionRef):
        return junc_map.get(att.id, (0, 0))
    spec = comp_map.get(att.component)
    if spec is None:
        return (0, 0)
    return (int(round(spec.center_px[0])), int(round(spec.center_px[1])))


def export_to_kicad_sch(
    doc: CircuitDocumentV2 | CircuitDocument,
    output_path: str | os.PathLike[str] | None = None,
    *,
    title: str = "Ohmbench Export",
    project_uuid: str | None = None,
) -> str:
    """Export an Ohmbench schematic document to a KiCad 7/8 `.kicad_sch` S-expression."""
    from circuitsim.wiring.serializer import migrate_v1_to_v2

    v2_doc = migrate_v1_to_v2(doc) if isinstance(doc, CircuitDocument) else doc
    root_uuid = project_uuid or _deterministic_uuid(f"root:{title}:{len(v2_doc.placed)}")

    comp_map = {p.component.name: p for p in v2_doc.placed}
    junc_map = {j.id: j.pos for j in v2_doc.graph.junctions}

    used_libs: list[str] = []
    seen_libs: set[str] = set()
    for p in v2_doc.placed:
        ctype = p.component.component_type
        lib_id = _LIB_MAP.get(ctype, (f"Ohmbench:{ctype}", "U"))[0]
        if lib_id not in seen_libs:
            seen_libs.add(lib_id)
            used_libs.append(lib_id)

    lines: list[str] = [
        "(kicad_sch",
        "  (version 20231120)",
        '  (generator "ohmbench")',
        '  (generator_version "1.0")',
        f'  (uuid "{_esc(root_uuid)}")',
        '  (paper "A4")',
        "  (title_block",
        f'    (title "{_esc(title)}")',
        '    (comment 1 "Exported from Ohmbench")',
        "  )",
        "  (lib_symbols",
    ]
    for lib_id in used_libs:
        lines.append(_lib_symbol_sexpr(lib_id))
    lines.append("  )")

    # Junctions
    for j in v2_doc.graph.junctions:
        jx = _mm(j.pos[0])
        jy = _mm(j.pos[1])
        ju = _deterministic_uuid(f"junc:{j.id}")
        lines.append(
            f'  (junction (at {jx} {jy}) (diameter 0) (color 0 0 0 0) (uuid "{ju}"))'
        )
        j_lbl = str(getattr(j, "net_label", "") or "").strip().upper()
        if j_lbl:
            lu = _deterministic_uuid(f"label:junc:{j.id}:{j_lbl}")
            lines.append(
                f'  (label "{_esc(j_lbl)}" (at {jx} {jy} 0) '
                f'(effects (font (size 1.27 1.27)) (justify left bottom)) (uuid "{lu}"))'
            )

    # Orthogonal Manhattan wires
    for w in v2_doc.graph.wires:
        pa = _attachment_point_px(w.end_a, comp_map, junc_map)
        pb = _attachment_point_px(w.end_b, comp_map, junc_map)
        pts = [pa, *w.hints, pb]
        # Expand diagonal hops into orthogonal L-segments cleanly
        ortho: list[tuple[float, float]] = [(_mm(pts[0][0]), _mm(pts[0][1]))]
        for pt in pts[1:]:
            nx, ny = _mm(pt[0]), _mm(pt[1])
            px_prev, py_prev = ortho[-1]
            if nx != px_prev and ny != py_prev:
                ortho.append((nx, py_prev))
            if (nx, ny) != ortho[-1]:
                ortho.append((nx, ny))
        for idx in range(len(ortho) - 1):
            x1, y1 = ortho[idx]
            x2, y2 = ortho[idx + 1]
            if x1 == x2 and y1 == y2:
                continue
            wu = _deterministic_uuid(f"wire:{w.id}:{idx}")
            lines.append(
                f"  (wire (pts (xy {x1} {y1}) (xy {x2} {y2})) "
                f'(stroke (width 0) (type default)) (uuid "{wu}"))'
            )
        w_lbl = str(getattr(w, "net_label", "") or "").strip().upper()
        if w_lbl:
            mx = round(0.5 * (_mm(pa[0]) + _mm(pb[0])), 3)
            my = round(0.5 * (_mm(pa[1]) + _mm(pb[1])), 3)
            lu = _deterministic_uuid(f"label:wire:{w.id}:{w_lbl}")
            lines.append(
                f'  (label "{_esc(w_lbl)}" (at {mx} {my} 0) '
                f'(effects (font (size 1.27 1.27)) (justify left bottom)) (uuid "{lu}"))'
            )

    # Placed component symbols
    for p in v2_doc.placed:
        comp = p.component
        ctype = comp.component_type
        lib_id = _LIB_MAP.get(ctype, (f"Ohmbench:{ctype}", "U"))[0]
        cx = _mm(p.center_px[0])
        cy = _mm(p.center_px[1])
        ang = int(p.rotation_deg) % 360
        su = _deterministic_uuid(f"sym:{comp.name}")
        val = _display_value(p)
        lines.extend(
            [
                f'  (symbol (lib_id "{_esc(lib_id)}") (at {cx} {cy} {ang}) (unit 1)',
                f'    (in_bom {"no" if ctype == "ground" else "yes"}) (on_board {"no" if ctype == "ground" else "yes"}) (dnp no)',
                f'    (uuid "{su}")',
                f'    (property "Reference" "{_esc(comp.name)}" (at {cx} {round(cy - 2.54, 3)} 0) (effects (font (size 1.27 1.27))))',
                f'    (property "Value" "{_esc(val)}" (at {cx} {round(cy + 2.54, 3)} 0) (effects (font (size 1.27 1.27))))',
                f'    (property "Footprint" "" (at {cx} {cy} 0) (effects (font (size 1.27 1.27)) hide))',
                f'    (property "Datasheet" "~" (at {cx} {cy} 0) (effects (font (size 1.27 1.27)) hide))',
                "  )",
            ]
        )

    lines.append(")")
    content = "\n".join(lines) + "\n"
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")
    return content

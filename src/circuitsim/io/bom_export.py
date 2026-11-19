"""Qt-free Bill of Materials (BOM) & Thermal Power Budget aggregator and CSV/Markdown exporters."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re
from typing import Iterable, Sequence

from circuitsim.components.base import Component
from circuitsim.components.ground import Ground
from circuitsim.dialogs.value_parser import format_engineering_value
from circuitsim.components.sources import (
    ACVoltageSource,
    DCCurrentSource,
    DCVoltageSource,
    WavVoltageSource,
)
from circuitsim.engine.stress import StressLevel, evaluate_component_stress


_PACKAGE_BY_TYPE: dict[str, str] = {
    "resistor": "AXIAL-1/4W",
    "capacitor": "CER-0805",
    "inductor": "RADIAL-L",
    "potentiometer": "TRIM-3P",
    "diode": "DO-35",
    "zener_diode": "DO-41",
    "led": "LED-5mm",
    "npn_bjt": "TO-92",
    "pnp_bjt": "TO-92",
    "nmos": "TO-92",
    "pmos": "TO-92",
    "opamp": "DIP-8",
    "comparator": "DIP-8",
    "ne555": "DIP-8",
    "mcu": "DIP-MCU",
    "d_flipflop": "74HC74",
    "jk_flipflop": "74LS73",
    "counter_4bit": "74HC161",
    "seven_seg_display": "7SEG-HEX",
    "transformer": "XFMR-4P",
    "spdt_relay": "RELAY-5P",
    "fuse": "FUSE-5x20",
    "switch": "SPST",
    "lightbulb": "LAMP",
}

_SOURCE_TYPES: tuple[type[Component], ...] = (
    DCVoltageSource,
    ACVoltageSource,
    DCCurrentSource,
    WavVoltageSource,
)


def _natural_key(name: str) -> tuple[str, int, str]:
    m = re.match(r"^([A-Za-z_]+)(\d+)(.*)$", name)
    if m:
        return (m.group(1), int(m.group(2)), m.group(3))
    return (name, 0, "")


@dataclass(frozen=True)
class BOMLineItem:
    """Single grouped Bill of Materials line item with thermal power summary."""

    component_type: str
    value: str
    package: str
    quantity: int
    designators: tuple[str, ...]
    unit_power_mw: float
    total_power_mw: float
    max_power_mw: float
    max_stress_percent: float


@dataclass(frozen=True)
class BOMReport:
    """Complete Bill of Materials and passive thermal power budget report."""

    items: tuple[BOMLineItem, ...]
    total_components: int
    total_power_mw: float
    peak_stress_designator: str
    peak_stress_percent: float


def _component_package(comp: Component) -> str:
    if comp.component_type == "capacitor" and getattr(comp, "polarized", False):
        return "ELEC-RAD"
    return _PACKAGE_BY_TYPE.get(comp.component_type, comp.component_type.upper())


def _component_dissipated_mw(comp: Component) -> float:
    if isinstance(comp, _SOURCE_TYPES):
        return 0.0
    if comp.component_type == "resistor":
        r = float(getattr(comp, "resistance", 0.0))
        i = float(getattr(comp, "current", 0.0))
        if r > 0.0:
            return (i * i * r) * 1000.0
    return abs(float(getattr(comp, "power", 0.0))) * 1000.0


def _component_value_text(comp: Component) -> str:
    disp = str(getattr(comp, "display_value", "") or "").strip()
    if disp:
        return disp
    params = comp.get_editable_parameters()
    if params:
        p0 = params[0]
        val = comp.get_parameter_value(p0.key)
        return format_engineering_value(val, p0.unit)
    return comp.component_type


def generate_bom_report(
    circuit_or_components: object,
    solve_result: object | None = None,
) -> BOMReport:
    """Aggregate components into a grouped BOM + thermal power budget report."""
    if hasattr(circuit_or_components, "components"):
        raw_comps: Iterable[Component] = getattr(circuit_or_components, "components")
        circ_obj = circuit_or_components
    else:
        raw_comps = circuit_or_components  # type: ignore[assignment]
        circ_obj = None

    comp_list = [c for c in raw_comps if not isinstance(c, Ground)]
    stress_map = (
        evaluate_component_stress(circ_obj if circ_obj is not None else comp_list, solve_result)
        if comp_list
        else {}
    )

    grouped: dict[tuple[str, str, str], list[Component]] = {}
    for comp in comp_list:
        val_str = _component_value_text(comp)
        pkg_str = _component_package(comp)
        key = (comp.component_type, val_str, pkg_str)
        grouped.setdefault(key, []).append(comp)

    line_items: list[BOMLineItem] = []
    total_power_mw = 0.0
    peak_des = ""
    peak_pct = 0.0

    for (ctype, val_str, pkg_str), members in sorted(grouped.items()):
        members_sorted = sorted(members, key=lambda c: _natural_key(c.name))
        des = tuple(c.name for c in members_sorted)
        powers_mw = [_component_dissipated_mw(c) for c in members_sorted]
        grp_power_mw = sum(powers_mw)
        unit_power_mw = grp_power_mw / max(1, len(members_sorted))
        total_power_mw += grp_power_mw

        max_p_mw = max(float(getattr(c, "max_power", 0.25)) * 1000.0 for c in members_sorted)
        grp_stress = 0.0
        for c in members_sorted:
            rep = stress_map.get(c.name)
            pct = float(rep.utilization_pct) if rep is not None and rep.level != StressLevel.NORMAL or (rep is not None) else 0.0
            if pct > grp_stress:
                grp_stress = pct
            if pct >= peak_pct:
                peak_pct = pct
                peak_des = c.name

        line_items.append(
            BOMLineItem(
                component_type=ctype,
                value=val_str,
                package=pkg_str,
                quantity=len(members_sorted),
                designators=des,
                unit_power_mw=unit_power_mw,
                total_power_mw=grp_power_mw,
                max_power_mw=max_p_mw,
                max_stress_percent=grp_stress,
            )
        )

    return BOMReport(
        items=tuple(line_items),
        total_components=len(comp_list),
        total_power_mw=total_power_mw,
        peak_stress_designator=peak_des,
        peak_stress_percent=peak_pct,
    )


def export_bom_csv(report: BOMReport, path: str | Path | None = None) -> str:
    """Format a ``BOMReport`` as CSV (optionally writing to ``path``)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Type",
            "Value",
            "Package",
            "Qty",
            "Designators",
            "Avg Power (mW)",
            "Total Power (mW)",
            "Rating (mW)",
            "Max Stress (%)",
        ]
    )
    for item in report.items:
        writer.writerow(
            [
                item.component_type,
                item.value,
                item.package,
                item.quantity,
                ", ".join(item.designators),
                f"{item.unit_power_mw:.3f}",
                f"{item.total_power_mw:.3f}",
                f"{item.max_power_mw:.2f}",
                f"{item.max_stress_percent:.1f}",
            ]
        )
    text = buf.getvalue()
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def export_bom_markdown(report: BOMReport, path: str | Path | None = None) -> str:
    """Format a ``BOMReport`` as a GitHub-flavored Markdown BOM & thermal budget table."""
    lines = [
        "# Bill of Materials & Thermal Power Budget",
        "",
        f"- **Total Parts:** {report.total_components}",
        f"- **Total Dissipated Power:** {report.total_power_mw:.3f} mW",
        (
            f"- **Peak Thermal Stress:** {report.peak_stress_designator} ({report.peak_stress_percent:.1f}%)"
            if report.peak_stress_designator
            else "- **Peak Thermal Stress:** 0.0%"
        ),
        "",
        "| Type | Value | Package | Qty | Designators | Power (mW) | Rating (mW) | Max Stress (%) |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in report.items:
        lines.append(
            f"| {item.component_type} | {item.value} | {item.package} | {item.quantity} | "
            f"{', '.join(item.designators)} | {item.total_power_mw:.3f} | "
            f"{item.max_power_mw:.1f} | {item.max_stress_percent:.1f}% |"
        )
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text

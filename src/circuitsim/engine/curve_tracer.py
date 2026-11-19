"""Qt-free I-V Characteristic Curve Tracer engine for two-terminal and transistor devices."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Sequence
import numpy as np

from circuitsim.components.base import Component
from circuitsim.components.bjt import BJTBase, PNPBJT
from circuitsim.components.ground import Ground
from circuitsim.components.mosfet import MOSFETBase
from circuitsim.components.sources import DCCurrentSource, DCVoltageSource
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.solver import MNASolver


@dataclass(frozen=True)
class IVCurve:
    """Single I-V trace sweep curve."""

    label: str
    control_value: float
    voltages: list[float]
    currents_ma: list[float]


@dataclass(frozen=True)
class CurveFamilyResult:
    """Collection of I-V curves for one device across optional stepped bias parameters."""

    component_type: str
    x_label: str
    y_label: str
    control_label: str
    curves: list[IVCurve]


def trace_two_terminal_iv(
    comp: Component,
    v_start: float,
    v_stop: float,
    points: int = 50,
) -> IVCurve:
    """Sweep voltage `v_start..v_stop` across a cloned 2-terminal device and record current in mA."""
    if len(comp.nodes) < 2:
        raise ValueError(f"Component {comp.name!r} must have at least 2 terminals for 2-pin I-V trace.")
    n_pts = max(2, int(points))
    dut = copy.deepcopy(comp)
    dut.reset_state()
    if len(dut.nodes) == 2:
        dut.nodes = ("n_dut", "0")
    else:
        dut.nodes = ("n_dut", "0", *dut.nodes[2:])

    v_src = DCVoltageSource("V_SWEEP", "n_dut", "0", voltage=float(v_start))
    circ = Circuit()
    circ.add(v_src)
    circ.add(dut)
    circ.add(Ground("G_SWEEP", "0"))
    solver = MNASolver(circ)

    v_grid = np.linspace(float(v_start), float(v_stop), n_pts)
    voltages: list[float] = []
    currents_ma: list[float] = []
    for v_val in v_grid:
        v_f = float(v_val)
        v_src.dc_voltage = v_f
        res = solver.solve_dc()
        voltages.append(v_f)
        currents_ma.append(float(res.branch_currents.get(dut.name, dut.current)) * 1000.0)

    return IVCurve(
        label=f"{comp.name} ({comp.component_type})",
        control_value=0.0,
        voltages=voltages,
        currents_ma=currents_ma,
    )


def trace_transistor_family(
    comp: Component,
    v_main_start: float,
    v_main_stop: float,
    control_values: Sequence[float],
    points: int = 40,
) -> CurveFamilyResult:
    """Trace multi-curve BJT (I_C vs V_CE @ I_B) or MOSFET (I_D vs V_DS @ V_GS) characteristic family."""
    n_pts = max(2, int(points))
    v_grid = np.linspace(float(v_main_start), float(v_main_stop), n_pts)
    curves: list[IVCurve] = []

    if isinstance(comp, BJTBase):
        is_pnp = isinstance(comp, PNPBJT)
        for ib_raw in control_values:
            ib_val = -abs(float(ib_raw)) if (is_pnp and ib_raw > 0) else float(ib_raw)
            dut = copy.deepcopy(comp)
            dut.reset_state()
            dut.nodes = ("n_b", "n_c", "0")
            ib_src = DCCurrentSource("I_B_STEP", "n_b", "0", current=ib_val)
            vce_src = DCVoltageSource("V_CE_SWEEP", "n_c", "0", voltage=float(v_main_start))
            circ = Circuit()
            circ.add(ib_src)
            circ.add(vce_src)
            circ.add(dut)
            circ.add(Ground("G_TRACE", "0"))
            solver = MNASolver(circ)

            xs: list[float] = []
            ys: list[float] = []
            for v_ce in v_grid:
                v_f = float(v_ce)
                vce_src.dc_voltage = v_f
                solver.solve_dc()
                xs.append(v_f)
                ys.append(float(dut.i_c) * 1000.0)

            curves.append(
                IVCurve(
                    label=f"I_B = {float(ib_raw) * 1e6:g} uA",
                    control_value=float(ib_raw),
                    voltages=xs,
                    currents_ma=ys,
                )
            )
        return CurveFamilyResult(
            component_type=comp.component_type,
            x_label="V_CE (V)",
            y_label="I_C (mA)",
            control_label="I_B (A)",
            curves=curves,
        )

    if isinstance(comp, MOSFETBase):
        for vgs_val in control_values:
            dut = copy.deepcopy(comp)
            dut.reset_state()
            dut.nodes = ("n_g", "n_d", "0")
            vgs_src = DCVoltageSource("V_GS_STEP", "n_g", "0", voltage=float(vgs_val))
            vds_src = DCVoltageSource("V_DS_SWEEP", "n_d", "0", voltage=float(v_main_start))
            circ = Circuit()
            circ.add(vgs_src)
            circ.add(vds_src)
            circ.add(dut)
            circ.add(Ground("G_TRACE", "0"))
            solver = MNASolver(circ)

            xs = []
            ys = []
            for v_ds in v_grid:
                v_f = float(v_ds)
                vds_src.dc_voltage = v_f
                solver.solve_dc()
                xs.append(v_f)
                ys.append(float(dut.i_d) * 1000.0)

            curves.append(
                IVCurve(
                    label=f"V_GS = {float(vgs_val):g} V",
                    control_value=float(vgs_val),
                    voltages=xs,
                    currents_ma=ys,
                )
            )
        return CurveFamilyResult(
            component_type=comp.component_type,
            x_label="V_DS (V)",
            y_label="I_D (mA)",
            control_label="V_GS (V)",
            curves=curves,
        )

    raise TypeError(
        f"trace_transistor_family expects a BJT or MOSFET component, got {comp.component_type!r}."
    )


def export_curve_family_csv(
    result: CurveFamilyResult | IVCurve,
    path: str | Path | None = None,
) -> str:
    """Serialize an `IVCurve` or `CurveFamilyResult` to CSV text (and write to disk if `path` given)."""
    if isinstance(result, IVCurve):
        fam = CurveFamilyResult(
            component_type="two_terminal",
            x_label="V (V)",
            y_label="I (mA)",
            control_label="step",
            curves=[result],
        )
    else:
        fam = result

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Curve", fam.control_label, fam.x_label, fam.y_label])
    for curve in fam.curves:
        for v_val, i_ma in zip(curve.voltages, curve.currents_ma):
            writer.writerow([curve.label, f"{curve.control_value:.6g}", f"{v_val:.6g}", f"{i_ma:.6g}"])

    csv_text = buf.getvalue()
    if path is not None:
        Path(path).write_text(csv_text, encoding="utf-8")
    return csv_text

"""Operational Amplifier (Op-Amp) with smooth tanh rail saturation (no Qt)."""

from __future__ import annotations

import math
from typing import Any

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class OpAmp(Component):
    """3-terminal Operational Amplifier (`in_pos`, `in_neg`, `out`) with soft rail clamping.

    Node ordering matches ``opamp.json``:
        - ``nodes[0]``: ``in_pos`` (`+` non-inverting input)
        - ``nodes[1]``: ``in_neg`` (`-` inverting input)
        - ``nodes[2]``: ``out`` (output)
    """

    DEFAULT_AOL: float = 1e5
    DEFAULT_RIN: float = 10e6
    DEFAULT_ROUT: float = 10.0
    DEFAULT_VRAIL_POS: float = 15.0
    DEFAULT_VRAIL_NEG: float = -15.0

    def __init__(
        self,
        name: str,
        node_in_pos: str,
        node_in_neg: str,
        node_out: str,
        *,
        a_ol: float | str = DEFAULT_AOL,
        r_in: float | str = DEFAULT_RIN,
        r_out: float | str = DEFAULT_ROUT,
        v_rail_pos: float | str = DEFAULT_VRAIL_POS,
        v_rail_neg: float | str = DEFAULT_VRAIL_NEG,
        model_name: str = "IdealOpAmp",
        label: str | None = None,
        show_value: bool = True,
        **spice_aliases: Any,
    ) -> None:
        raw_aol = spice_aliases.pop("AOL", a_ol)
        raw_rin = spice_aliases.pop("RIN", r_in)
        raw_rout = spice_aliases.pop("ROUT", r_out)
        if spice_aliases:
            unknown = ", ".join(sorted(spice_aliases))
            raise TypeError(f"OpAmp {name!r} got unexpected keyword argument(s): {unknown}")

        val_aol = parse_engineering_value(raw_aol)
        val_rin = parse_engineering_value(raw_rin)
        val_rout = parse_engineering_value(raw_rout)
        val_vp = parse_engineering_value(v_rail_pos)
        val_vn = parse_engineering_value(v_rail_neg)

        if val_aol <= 0.0:
            raise ValueError(f"OpAmp {name!r}: a_ol must be > 0 (got {val_aol}).")
        if val_rin <= 0.0:
            raise ValueError(f"OpAmp {name!r}: r_in must be > 0 ohm (got {val_rin}).")
        if val_rout <= 0.0:
            raise ValueError(f"OpAmp {name!r}: r_out must be > 0 ohm (got {val_rout}).")
        if val_vp <= val_vn:
            raise ValueError(
                f"OpAmp {name!r}: v_rail_pos ({val_vp}) must be > v_rail_neg ({val_vn})."
            )

        super().__init__(
            name,
            (node_in_pos, node_in_neg, node_out),
            label=label,
            show_value=show_value,
        )
        self.a_ol: float = val_aol
        self.r_in: float = val_rin
        self.r_out: float = val_rout
        self.v_rail_pos: float = val_vp
        self.v_rail_neg: float = val_vn
        self.model_name: str = model_name
        self.operating_vd: float = 0.0
        self.set_max_ratings(max_power=0.5, max_voltage=36.0, max_current=0.1)

    @property
    def node_in_pos(self) -> str:
        return self.nodes[0]

    @property
    def node_in_neg(self) -> str:
        return self.nodes[1]

    @property
    def node_out(self) -> str:
        return self.nodes[2]

    @property
    def component_type(self) -> str:
        return "opamp"

    @property
    def display_value(self) -> str:
        return f"A={self.a_ol:.0e}"

    def eval_transfer(self, v_d: float) -> tuple[float, float]:
        """Return `(v_target, dv_target_dvd)` with smooth `tanh` rail clamping."""
        v_mid = 0.5 * (self.v_rail_pos + self.v_rail_neg)
        v_half = max(1e-6, 0.5 * (self.v_rail_pos - self.v_rail_neg))
        u = (self.a_ol * v_d - v_mid) / v_half
        u_clamped = max(-40.0, min(40.0, u))
        th = math.tanh(u_clamped)
        v_target = v_mid + v_half * th
        slope = max(1e-12, self.a_ol * (1.0 - th * th))
        return (v_target, slope)

    def limit_diff_voltage(self, vd_new: float, vd_old: float) -> tuple[float, bool]:
        """Damp large jumps across the linear gain region to preserve monotonic NR convergence."""
        v_half = max(1e-6, 0.5 * (self.v_rail_pos - self.v_rail_neg))
        linear_bw = max(1e-4, 4.0 * v_half / self.a_ol)
        dv = vd_new - vd_old
        if abs(dv) > max(0.25, 10.0 * linear_bw) and abs(vd_old) <= 2.0 * linear_bw:
            return (vd_old + 0.5 * dv, True)
        return (vd_new, False)

    def companion_stamp(
        self, v_d: float | None = None
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
        """Return 3x3 Jacobian matrix and 3-element RHS vector for `(in_pos, in_neg, out)`."""
        vd0 = self.operating_vd if v_d is None else float(v_d)
        f0, a_eff = self.eval_transfer(vd0)
        g_in = 1.0 / self.r_in
        g_out = 1.0 / self.r_out
        g_m = g_out * a_eff

        g_mat = (
            (g_in, -g_in, 0.0),
            (-g_in, g_in, 0.0),
            (-g_m, g_m, g_out),
        )
        rhs_out = g_out * (f0 - a_eff * vd0)
        return (g_mat, (0.0, 0.0, rhs_out))

    def reset_state(self) -> None:
        super().reset_state()
        self.operating_vd = 0.0

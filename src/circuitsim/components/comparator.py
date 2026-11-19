"""Voltage Comparator macromodel with rail-to-rail tanh switching (no Qt)."""

from __future__ import annotations

import math
from typing import Any

from circuitsim.components.opamp import OpAmp
from circuitsim.dialogs.value_parser import parse_engineering_value


class Comparator(OpAmp):
    """3-terminal Voltage Comparator (`in_pos`, `in_neg`, `out`) with configurable logic rails.

    Subclasses `OpAmp` so it integrates cleanly into the damped Newton-Raphson solver,
    while clamping output smoothly between ``v_low`` and ``v_high`` around ``v_offset``.
    """

    DEFAULT_AOL: float = 1e5
    DEFAULT_RIN: float = 10e6
    DEFAULT_ROUT: float = 25.0
    DEFAULT_VHIGH: float = 5.0
    DEFAULT_VLOW: float = 0.0

    def __init__(
        self,
        name: str,
        node_in_pos: str,
        node_in_neg: str,
        node_out: str,
        *,
        v_high: float | str = DEFAULT_VHIGH,
        v_low: float | str = DEFAULT_VLOW,
        v_offset: float | str = 0.0,
        a_ol: float | str = DEFAULT_AOL,
        r_in: float | str = DEFAULT_RIN,
        r_out: float | str = DEFAULT_ROUT,
        model_name: str = "LM393",
        label: str | None = None,
        show_value: bool = True,
        **spice_aliases: Any,
    ) -> None:
        vh = parse_engineering_value(spice_aliases.pop("v_rail_pos", v_high))
        vl = parse_engineering_value(spice_aliases.pop("v_rail_neg", v_low))
        vos = parse_engineering_value(v_offset)
        super().__init__(
            name,
            node_in_pos,
            node_in_neg,
            node_out,
            a_ol=a_ol,
            r_in=r_in,
            r_out=r_out,
            v_rail_pos=vh,
            v_rail_neg=vl,
            model_name=model_name,
            label=label,
            show_value=show_value,
            **spice_aliases,
        )
        self.v_offset: float = vos

    @property
    def v_high(self) -> float:
        return self.v_rail_pos

    @v_high.setter
    def v_high(self, value: float) -> None:
        self.v_rail_pos = float(value)

    @property
    def v_low(self) -> float:
        return self.v_rail_neg

    @v_low.setter
    def v_low(self, value: float) -> None:
        self.v_rail_neg = float(value)

    @property
    def component_type(self) -> str:
        return "comparator"

    @property
    def display_value(self) -> str:
        return f"{self.v_low:g}..{self.v_high:g}V"

    def eval_transfer(self, v_d: float) -> tuple[float, float]:
        """Return `(v_target, dv_target_dvd)` with input offset ``v_offset``."""
        v_mid = 0.5 * (self.v_high + self.v_low)
        v_half = max(1e-6, 0.5 * (self.v_high - self.v_low))
        u = (self.a_ol * (v_d - self.v_offset)) / v_half
        u_clamped = max(-40.0, min(40.0, u))
        th = math.tanh(u_clamped)
        v_target = v_mid + v_half * th
        slope = max(1e-12, self.a_ol * (1.0 - th * th))
        return (v_target, slope)

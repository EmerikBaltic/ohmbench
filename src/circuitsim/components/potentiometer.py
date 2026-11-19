"""Three-terminal Potentiometer component model (no Qt dependencies)."""

from __future__ import annotations

import math

from circuitsim.components.base import Component, ComponentParameter
from circuitsim.dialogs.value_parser import parse_engineering_value


class Potentiometer(Component):
    """Three-terminal potentiometer modeled as two tracked series resistors.

    Terminals are ordered ``(node_a, node_w, node_b)``:
        - ``a``: top/left fixed end terminal
        - ``w``: movable wiper terminal
        - ``b``: bottom/right fixed end terminal

    With normalized wiper fraction ``alpha`` clamped to ``[0.001, 0.999]``:
        - ``r_wb = alpha * resistance`` (so ``v_w -> v_b`` as ``alpha -> 0``,
          and ``v_w -> v_a`` as ``alpha -> 1``; with ``v_b = 0`` and ``v_a = V``,
          ``v_w = alpha * V``).
        - ``r_aw = (1.0 - alpha) * resistance``.
    """

    MIN_ALPHA: float = 0.001
    MAX_ALPHA: float = 0.999
    MIN_RESISTANCE_OHM: float = 1e-3

    def __init__(
        self,
        name: str,
        node_a: str,
        node_w: str,
        node_b: str,
        *,
        resistance: float | str = 10_000.0,
        alpha: float = 0.5,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        super().__init__(
            name=name,
            nodes=(node_a, node_w, node_b),
            label=label,
            show_value=show_value,
            color=color,
        )
        parsed_r = parse_engineering_value(resistance)
        if not math.isfinite(parsed_r) or parsed_r <= 0.0:
            raise ValueError(f"Potentiometer {name!r}: resistance must be > 0 Ω.")
        alpha_f = float(alpha)
        if not math.isfinite(alpha_f):
            raise ValueError(f"Potentiometer {name!r}: alpha must be finite.")
        self.resistance: float = max(self.MIN_RESISTANCE_OHM, float(parsed_r))
        self.alpha: float = max(self.MIN_ALPHA, min(self.MAX_ALPHA, alpha_f))

    @property
    def component_type(self) -> str:
        return "potentiometer"

    @property
    def node_a(self) -> str:
        return self.nodes[0]

    @property
    def node_w(self) -> str:
        return self.nodes[1]

    @property
    def node_b(self) -> str:
        return self.nodes[2]

    @property
    def r_aw(self) -> float:
        return max(self.MIN_RESISTANCE_OHM * 1e-3, (1.0 - self.alpha) * self.resistance)

    @property
    def r_wb(self) -> float:
        return max(self.MIN_RESISTANCE_OHM * 1e-3, self.alpha * self.resistance)

    @property
    def g_aw(self) -> float:
        return 1.0 / self.r_aw

    @property
    def g_wb(self) -> float:
        return 1.0 / self.r_wb

    def set_alpha(self, alpha: float) -> float:
        alpha_f = float(alpha)
        if not math.isfinite(alpha_f):
            raise ValueError("alpha must be finite.")
        self.alpha = max(self.MIN_ALPHA, min(self.MAX_ALPHA, alpha_f))
        return self.alpha

    def set_resistance(self, value: float | str) -> float:
        parsed_r = parse_engineering_value(value)
        if not math.isfinite(parsed_r) or parsed_r <= 0.0:
            raise ValueError("Total resistance must be > 0 Ω.")
        self.resistance = max(self.MIN_RESISTANCE_OHM, float(parsed_r))
        return self.resistance

    def get_editable_parameters(self) -> list[ComponentParameter]:
        return [
            ComponentParameter(
                key="resistance",
                label="Total Resistance",
                unit="Ω",
                min_value=self.MIN_RESISTANCE_OHM,
                max_value=1e9,
                description="End-to-end track resistance between terminals a and b.",
            ),
            ComponentParameter(
                key="alpha",
                label="Wiper Position",
                unit="",
                min_value=self.MIN_ALPHA,
                max_value=self.MAX_ALPHA,
                description="Normalized wiper position in [0.001, 0.999] (0=b, 1=a).",
            ),
        ]

    def get_parameter_value(self, key: str) -> float:
        if key == "resistance":
            return self.resistance
        if key == "alpha":
            return self.alpha
        raise KeyError(f"Unknown parameter {key!r} for Potentiometer.")

    def set_parameter_value(self, key: str, value: float | str) -> float:
        if key == "resistance":
            return self.set_resistance(value)
        if key == "alpha":
            val = parse_engineering_value(value) if isinstance(value, str) else float(value)
            return self.set_alpha(val)
        raise KeyError(f"Unknown parameter {key!r} for Potentiometer.")

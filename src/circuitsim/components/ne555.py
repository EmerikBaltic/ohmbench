"""555 Precision Timer IC behavioral mixed-signal macromodel (no Qt)."""

from __future__ import annotations

from typing import Any, Mapping

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class NE555(Component):
    """NE555 Timer IC behavioral macromodel with 5k-5k-5k divider and SR latch.

    Terminal order (matches ``ne555.json``):
        0. ``vcc``   - Positive supply rail
        1. ``gnd``   - Ground reference pin
        2. ``trig``  - Trigger input (sets Q = 1 when V_trig < 1/3 V_cc)
        3. ``thr``   - Threshold input (resets Q = 0 when V_thr > 2/3 V_cc)
        4. ``dis``   - Open-collector discharge pin (low to GND when Q = 0)
        5. ``out``   - Push-pull output stage (VCC when Q = 1, GND when Q = 0)
        6. ``reset`` - Active-low reset input (forces Q = 0 when V_reset < v_reset_th)
        7. ``ctrl``  - Optional control voltage pin (2/3 V_cc node if wired)
    """

    def __init__(
        self,
        name: str,
        node_vcc: str | None = None,
        node_gnd: str | None = None,
        node_trig: str | None = None,
        node_thr: str | None = None,
        node_dis: str | None = None,
        node_out: str | None = None,
        node_reset: str | None = None,
        node_ctrl: str | None = None,
        *,
        r_div: float | str = 5000.0,
        r_out: float | str = 25.0,
        r_dis_on: float | str = 15.0,
        r_dis_off: float | str = 1e8,
        v_reset_th: float | str = 0.7,
        r_in: float | str = 1e7,
        initial_q: bool = False,
        label: str | None = None,
        show_value: bool = True,
    ) -> None:
        base_nodes = [
            node_vcc or f"{name}_vcc",
            node_gnd or f"{name}_gnd",
            node_trig or f"{name}_trig",
            node_thr or f"{name}_thr",
            node_dis or f"{name}_dis",
            node_out or f"{name}_out",
            node_reset or f"{name}_reset",
        ]
        if node_ctrl is not None:
            base_nodes.append(node_ctrl)
        super().__init__(
            name=name,
            nodes=tuple(base_nodes),
            label=label,
            show_value=show_value,
        )
        self.r_div: float = parse_engineering_value(r_div)
        self.r_out: float = parse_engineering_value(r_out)
        self.r_dis_on: float = parse_engineering_value(r_dis_on)
        self.r_dis_off: float = parse_engineering_value(r_dis_off)
        self.v_reset_th: float = parse_engineering_value(v_reset_th)
        self.r_in: float = parse_engineering_value(r_in)
        self.initial_q: bool = bool(initial_q)
        self.output_high: bool = bool(initial_q)

        if (
            self.r_div <= 0.0
            or self.r_out <= 0.0
            or self.r_dis_on <= 0.0
            or self.r_dis_off <= 0.0
            or self.r_in <= 0.0
        ):
            raise ValueError(f"NE555 {name!r}: all resistance parameters must be > 0 ohm.")
        self.set_max_ratings(max_power=0.6, max_voltage=18.0, max_current=0.2)

    @property
    def component_type(self) -> str:
        return "ne555"

    @property
    def display_value(self) -> str:
        state_lbl = "H" if self.output_high else "L"
        return f"NE555 ({state_lbl})"

    @property
    def node_vcc(self) -> str:
        return self.nodes[0]

    @property
    def node_gnd(self) -> str:
        return self.nodes[1]

    @property
    def node_trig(self) -> str:
        return self.nodes[2]

    @property
    def node_thr(self) -> str:
        return self.nodes[3]

    @property
    def node_dis(self) -> str:
        return self.nodes[4]

    @property
    def node_out(self) -> str:
        return self.nodes[5]

    @property
    def node_reset(self) -> str:
        return self.nodes[6]

    def reset_state(self) -> None:
        super().reset_state()
        self.output_high = self.initial_q

    def evaluate_latch(self, node_voltages: Mapping[str, float]) -> bool:
        """Update internal SR latch from solved terminal potentials; return True if Q flipped."""
        v_gnd = float(node_voltages.get(self.node_gnd, 0.0))
        v_cc = float(node_voltages.get(self.node_vcc, 0.0)) - v_gnd
        v_trig = float(node_voltages.get(self.node_trig, 0.0)) - v_gnd
        v_thr = float(node_voltages.get(self.node_thr, 0.0)) - v_gnd
        v_rst = float(node_voltages.get(self.node_reset, v_cc)) - v_gnd

        if len(self.nodes) >= 8:
            v_upper = float(node_voltages.get(self.nodes[7], (2.0 / 3.0) * v_cc)) - v_gnd
        else:
            v_upper = (2.0 / 3.0) * v_cc
        v_lower = 0.5 * v_upper

        if v_cc <= 0.5:
            new_q = False
        elif v_rst < self.v_reset_th:
            new_q = False
        elif v_trig < v_lower:
            new_q = True
        elif v_thr > v_upper:
            new_q = False
        else:
            new_q = self.output_high

        changed = new_q != self.output_high
        self.output_high = new_q
        return changed

    def companion_stamp(self) -> tuple[list[list[float]], list[float]]:
        """Return `(G_mat, rhs_vec)` across `self.nodes` for DC, AC, and transient MNA."""
        n = len(self.nodes)
        g_mat: list[list[float]] = [[0.0] * n for _ in range(n)]
        rhs: list[float] = [0.0] * n

        def _stamp(i: int, j: int, g: float) -> None:
            g_mat[i][i] += g
            g_mat[j][j] += g
            g_mat[i][j] -= g
            g_mat[j][i] -= g

        # Internal 5k / 5k / 5k resistor ladder between VCC (0) and GND (1)
        if n >= 8:
            _stamp(0, 7, 1.0 / self.r_div)
            _stamp(7, 1, 1.0 / (2.0 * self.r_div))
        else:
            _stamp(0, 1, 1.0 / (3.0 * self.r_div))

        # High-impedance comparator/reset inputs to GND
        g_in = 1.0 / self.r_in
        _stamp(2, 1, g_in)
        _stamp(3, 1, g_in)
        _stamp(6, 1, g_in)

        # Output stage drives OUT (5) toward VCC (0) or GND (1)
        g_out = 1.0 / self.r_out
        if self.output_high:
            _stamp(5, 0, g_out)
        else:
            _stamp(5, 1, g_out)

        # Discharge open-collector transistor pulls DIS (4) to GND (1) when Q = 0
        g_dis = (1.0 / self.r_dis_off) if self.output_high else (1.0 / self.r_dis_on)
        _stamp(4, 1, g_dis)

        return (g_mat, rhs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_div": self.r_div,
            "r_out": self.r_out,
            "r_dis_on": self.r_dis_on,
            "r_dis_off": self.r_dis_off,
            "v_reset_th": self.v_reset_th,
            "r_in": self.r_in,
            "initial_q": self.initial_q,
        }

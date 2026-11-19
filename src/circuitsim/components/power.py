"""Power & Protection components: Transformer, SPDT Relay, and Blowable/Resettable Fuse (no Qt)."""

from __future__ import annotations

import math
from typing import Any, Mapping

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class Transformer(Component):
    """Coupled 4-terminal Transformer (`p1`, `p2`, `s1`, `s2`) for DC, AC, and transient MNA.

    Models an ideal turns ratio ``n = turns_ratio * coupling_k`` with winding resistance
    ``r_winding_ohm`` and magnetizing inductance ``l_primary_h`` using a symmetric
    positive-definite 4x4 conductance stamp plus 1 GOhm primary-to-secondary isolation bleed
    so floating secondaries never trigger singular matrices.
    """

    def __init__(
        self,
        name: str,
        node_p1: str = "p1",
        node_p2: str = "p2",
        node_s1: str = "s1",
        node_s2: str = "s2",
        *,
        turns_ratio: float | str = 0.5,
        r_winding_ohm: float | str = 0.1,
        l_primary_h: float | str = 1.0,
        coupling_k: float | str = 0.999,
        label: str | None = None,
        show_value: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            nodes=(node_p1, node_p2, node_s1, node_s2),
            label=label,
            show_value=show_value,
        )
        self.turns_ratio: float = parse_engineering_value(turns_ratio)
        self.r_winding_ohm: float = parse_engineering_value(r_winding_ohm)
        self.l_primary_h: float = parse_engineering_value(l_primary_h)
        self.coupling_k: float = parse_engineering_value(coupling_k)
        if self.turns_ratio <= 0.0:
            raise ValueError(f"Transformer {name!r}: turns_ratio must be > 0.")
        if self.r_winding_ohm <= 0.0:
            raise ValueError(f"Transformer {name!r}: r_winding_ohm must be > 0.")
        if self.l_primary_h <= 0.0:
            raise ValueError(f"Transformer {name!r}: l_primary_h must be > 0.")
        if not (0.0 < self.coupling_k <= 1.0):
            raise ValueError(f"Transformer {name!r}: coupling_k must be in (0, 1].")
        self.set_max_ratings(max_power=50.0, max_voltage=250.0, max_current=5.0)

    @property
    def component_type(self) -> str:
        return "transformer"

    @property
    def display_value(self) -> str:
        ratio = self.turns_ratio
        if ratio < 1.0 and ratio > 0.0:
            return f"1:{1.0 / ratio:g}"
        return f"1:{ratio:g}" if abs(ratio - 1.0) < 1e-9 else f"n={ratio:g}"

    @property
    def node_p1(self) -> str:
        return self.nodes[0]

    @property
    def node_p2(self) -> str:
        return self.nodes[1]

    @property
    def node_s1(self) -> str:
        return self.nodes[2]

    @property
    def node_s2(self) -> str:
        return self.nodes[3]

    def companion_stamp(self, freq_hz: float = 50.0) -> tuple[list[list[float]], list[float]]:
        """Return symmetric 4x4 `(G_mat, rhs)` across `(p1, p2, s1, s2)`."""
        n = self.turns_ratio * self.coupling_k
        g_w = 1.0 / max(self.r_winding_ohm, 1e-4)
        g_mag = 1.0 / max(1000.0, 2.0 * math.pi * max(1.0, freq_hz) * self.l_primary_h)
        g_iso = 1e-9

        g_pp = g_mag + (n * n) * g_w
        g_ps = -n * g_w
        g_ss = g_w

        g_mat = [
            [g_pp, -g_pp, g_ps, -g_ps],
            [-g_pp, g_pp + g_iso, -g_ps, g_ps - g_iso],
            [g_ps, -g_ps, g_ss, -g_ss],
            [-g_ps, g_ps - g_iso, -g_ss, g_ss + g_iso],
        ]
        return (g_mat, [0.0, 0.0, 0.0, 0.0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns_ratio": self.turns_ratio,
            "r_winding_ohm": self.r_winding_ohm,
            "l_primary_h": self.l_primary_h,
            "coupling_k": self.coupling_k,
        }


class SPDTRelay(Component):
    """5-terminal SPDT Electromechanical Relay (`coil_pos`, `coil_neg`, `com`, `no`, `nc`) with hysteresis."""

    def __init__(
        self,
        name: str,
        node_coil_pos: str = "coil_pos",
        node_coil_neg: str = "coil_neg",
        node_com: str = "com",
        node_no: str = "no",
        node_nc: str = "nc",
        *,
        coil_resistance_ohm: float | str = 100.0,
        pull_in_voltage_v: float | str = 5.0,
        drop_out_voltage_v: float | str = 2.5,
        r_on: float | str = 0.01,
        r_off: float | str = 1e9,
        energized: bool = False,
        label: str | None = None,
        show_value: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            nodes=(node_coil_pos, node_coil_neg, node_com, node_no, node_nc),
            label=label,
            show_value=show_value,
        )
        self.coil_resistance_ohm: float = parse_engineering_value(coil_resistance_ohm)
        self.pull_in_voltage_v: float = parse_engineering_value(pull_in_voltage_v)
        self.drop_out_voltage_v: float = parse_engineering_value(drop_out_voltage_v)
        self.r_on: float = parse_engineering_value(r_on)
        self.r_off: float = parse_engineering_value(r_off)
        self.energized: bool = bool(energized)
        self._initial_energized: bool = bool(energized)
        if self.coil_resistance_ohm <= 0.0 or self.r_on <= 0.0 or self.r_off <= 0.0:
            raise ValueError(f"SPDTRelay {name!r}: resistances must be > 0 ohm.")
        if self.pull_in_voltage_v <= 0.0 or self.drop_out_voltage_v < 0.0:
            raise ValueError(f"SPDTRelay {name!r}: invalid coil threshold voltages.")
        self.set_max_ratings(max_power=2.0, max_voltage=250.0, max_current=10.0)

    @property
    def component_type(self) -> str:
        return "spdt_relay"

    @property
    def display_value(self) -> str:
        state_lbl = "NO" if self.energized else "NC"
        return f"{self.pull_in_voltage_v:g}V ({state_lbl})"

    @property
    def node_coil_pos(self) -> str:
        return self.nodes[0]

    @property
    def node_coil_neg(self) -> str:
        return self.nodes[1]

    @property
    def node_com(self) -> str:
        return self.nodes[2]

    @property
    def node_no(self) -> str:
        return self.nodes[3]

    @property
    def node_nc(self) -> str:
        return self.nodes[4]

    def reset_state(self) -> None:
        super().reset_state()
        self.energized = self._initial_energized

    def evaluate_coil(self, node_voltages: Mapping[str, float]) -> bool:
        """Update contact state with pull-in / drop-out hysteresis; return True if toggled."""
        v_coil = abs(
            float(node_voltages.get(self.node_coil_pos, 0.0))
            - float(node_voltages.get(self.node_coil_neg, 0.0))
        )
        new_state = self.energized
        if not self.energized and v_coil >= self.pull_in_voltage_v:
            new_state = True
        elif self.energized and v_coil <= self.drop_out_voltage_v:
            new_state = False
        changed = new_state != self.energized
        self.energized = new_state
        return changed

    def companion_stamp(self) -> tuple[list[list[float]], list[float]]:
        """Return 5x5 `(G_mat, rhs)` across `(coil_pos, coil_neg, com, no, nc)`."""
        g_mat: list[list[float]] = [[0.0] * 5 for _ in range(5)]

        def _stamp(i: int, j: int, g: float) -> None:
            g_mat[i][i] += g
            g_mat[j][j] += g
            g_mat[i][j] -= g
            g_mat[j][i] -= g

        _stamp(0, 1, 1.0 / self.coil_resistance_ohm)
        g_no = (1.0 / self.r_on) if self.energized else (1.0 / self.r_off)
        g_nc = (1.0 / self.r_off) if self.energized else (1.0 / self.r_on)
        _stamp(2, 3, g_no)
        _stamp(2, 4, g_nc)
        return (g_mat, [0.0, 0.0, 0.0, 0.0, 0.0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "coil_resistance_ohm": self.coil_resistance_ohm,
            "pull_in_voltage_v": self.pull_in_voltage_v,
            "drop_out_voltage_v": self.drop_out_voltage_v,
            "r_on": self.r_on,
            "r_off": self.r_off,
            "energized": self.energized,
        }


class Fuse(Component):
    """2-terminal Blowable & Resettable Overcurrent Fuse (`a`, `b`)."""

    def __init__(
        self,
        name: str,
        node_a: str = "a",
        node_b: str = "b",
        *,
        rating_amps: float | str = 0.5,
        r_cold_ohm: float | str = 0.05,
        r_blown_ohm: float | str = 1e9,
        is_blown: bool = False,
        label: str | None = None,
        show_value: bool = True,
    ) -> None:
        super().__init__(name=name, nodes=(node_a, node_b), label=label, show_value=show_value)
        self.rating_amps: float = parse_engineering_value(rating_amps)
        self.r_cold_ohm: float = parse_engineering_value(r_cold_ohm)
        self.r_blown_ohm: float = parse_engineering_value(r_blown_ohm)
        self.is_blown: bool = bool(is_blown)
        if self.rating_amps <= 0.0:
            raise ValueError(f"Fuse {name!r}: rating_amps must be > 0 A.")
        if self.r_cold_ohm <= 0.0 or self.r_blown_ohm <= 0.0:
            raise ValueError(f"Fuse {name!r}: resistances must be > 0 ohm.")
        self.set_max_ratings(max_power=5.0, max_voltage=250.0, max_current=self.rating_amps)

    @property
    def component_type(self) -> str:
        return "fuse"

    @property
    def conductance(self) -> float:
        return (1.0 / self.r_blown_ohm) if self.is_blown else (1.0 / self.r_cold_ohm)

    @property
    def resistance(self) -> float:
        return self.r_blown_ohm if self.is_blown else self.r_cold_ohm

    @property
    def display_value(self) -> str:
        if self.is_blown:
            return f"BLOWN ({self.rating_amps:g}A)"
        return f"{self.rating_amps:g}A"

    def check_blow(self, current_amps: float) -> bool:
        """Open fuse permanently (until reset) if ``|current_amps| > rating_amps``."""
        if not self.is_blown and abs(current_amps) > self.rating_amps * (1.0 + 1e-6):
            self.is_blown = True
            return True
        return False

    def reset_fuse(self) -> None:
        """Restore intact low-resistance fuse conduction."""
        self.is_blown = False

    def reset_state(self) -> None:
        super().reset_state()
        self.is_blown = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rating_amps": self.rating_amps,
            "r_cold_ohm": self.r_cold_ohm,
            "r_blown_ohm": self.r_blown_ohm,
            "is_blown": self.is_blown,
        }

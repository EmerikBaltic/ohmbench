"""Power & Protection components: Transformer, SPDT Relay, and Blowable/Resettable Fuse (no Qt)."""

from __future__ import annotations

import math
from typing import Any, Mapping

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class Transformer(Component):
    """Coupled 4-terminal Transformer (`p1`, `p2`, `s1`, `s2`) for DC, AC, and transient MNA.

    Supports both ``Ideal`` (lossless/near-ideal k~1) and ``Non-Ideal`` (finite magnetizing
    inductance ``l_primary_h``, magnetic coupling ``coupling_k``, and primary/secondary winding
    resistances ``r_primary_ohm`` / ``r_secondary_ohm``) modes using a magnetically coupled
    2-port RL companion model in transient/AC and DC-blocking winding resistances in ``solve_dc``.
    """

    def __init__(
        self,
        name: str,
        node_p1: str = "p1",
        node_p2: str = "p2",
        node_s1: str = "s1",
        node_s2: str = "s2",
        *,
        primary_turns: float | str = 100.0,
        secondary_turns: float | str | None = None,
        turns_ratio: float | str | None = None,
        is_ideal: bool = True,
        l_primary_h: float | str = 1.0,
        coupling_k: float | str = 0.98,
        r_primary_ohm: float | str = 1.0,
        r_secondary_ohm: float | str = 1.0,
        r_winding_ohm: float | str | None = None,
        label: str | None = None,
        show_value: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            nodes=(node_p1, node_p2, node_s1, node_s2),
            label=label,
            show_value=show_value,
        )
        self.primary_turns: float = parse_engineering_value(primary_turns)
        if self.primary_turns < 1.0:
            raise ValueError(f"Transformer {name!r}: primary_turns must be >= 1.")
        if secondary_turns is not None:
            self.secondary_turns: float = parse_engineering_value(secondary_turns)
            if self.secondary_turns < 1.0:
                raise ValueError(f"Transformer {name!r}: secondary_turns must be >= 1.")
        elif turns_ratio is not None:
            ratio_val = parse_engineering_value(turns_ratio)
            if ratio_val <= 0.0:
                raise ValueError(f"Transformer {name!r}: turns_ratio must be > 0.")
            self.secondary_turns = max(1e-6, self.primary_turns * ratio_val)
        else:
            self.secondary_turns = 200.0

        self.is_ideal: bool = bool(is_ideal)
        self.l_primary_h: float = parse_engineering_value(l_primary_h)
        self.coupling_k: float = parse_engineering_value(coupling_k)
        if r_winding_ohm is not None:
            rw = parse_engineering_value(r_winding_ohm)
            self.r_primary_ohm: float = rw
            self.r_secondary_ohm: float = rw
        else:
            self.r_primary_ohm = parse_engineering_value(r_primary_ohm)
            self.r_secondary_ohm = parse_engineering_value(r_secondary_ohm)

        if self.r_primary_ohm <= 0.0 or self.r_secondary_ohm <= 0.0:
            raise ValueError(f"Transformer {name!r}: winding resistances must be > 0.")
        if self.l_primary_h <= 0.0:
            raise ValueError(f"Transformer {name!r}: l_primary_h must be > 0.")
        if not (0.0 < self.coupling_k <= 1.0):
            raise ValueError(f"Transformer {name!r}: coupling_k must be in (0, 1].")

        self.i_p: float = 0.0
        self.i_s: float = 0.0
        self.v_ind_p: float = 0.0
        self.v_ind_s: float = 0.0
        self.set_max_ratings(max_power=50.0, max_voltage=250.0, max_current=5.0)

    @property
    def component_type(self) -> str:
        return "transformer"

    @property
    def turns_ratio(self) -> float:
        return self.secondary_turns / max(self.primary_turns, 1e-9)

    @turns_ratio.setter
    def turns_ratio(self, val: float) -> None:
        v = float(val)
        if v <= 0.0:
            raise ValueError(f"Transformer {self.name!r}: turns_ratio must be > 0.")
        self.secondary_turns = max(1e-6, self.primary_turns * v)

    @property
    def r_winding_ohm(self) -> float:
        return self.r_primary_ohm

    @r_winding_ohm.setter
    def r_winding_ohm(self, val: float) -> None:
        v = float(val)
        if v <= 0.0:
            raise ValueError(f"Transformer {self.name!r}: winding resistance must be > 0.")
        self.r_primary_ohm = v
        self.r_secondary_ohm = v

    @property
    def display_value(self) -> str:
        mode_lbl = "Ideal" if self.is_ideal else "Real"
        return f"{self.primary_turns:g}:{self.secondary_turns:g} ({mode_lbl})"

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

    def reset_state(self) -> None:
        super().reset_state()
        self.i_p = 0.0
        self.i_s = 0.0
        self.v_ind_p = 0.0
        self.v_ind_s = 0.0

    def set_parameter_value(self, key: str, raw_value: float | str) -> float:
        if key == "is_ideal":
            self.is_ideal = bool(raw_value) if not isinstance(raw_value, str) else (raw_value.strip().lower() in ("1", "true", "yes"))
            return 1.0 if self.is_ideal else 0.0
        if key == "turns_ratio":
            val = parse_engineering_value(raw_value)
            self.turns_ratio = val
            return val
        if key == "r_winding_ohm":
            val = parse_engineering_value(raw_value)
            self.r_winding_ohm = val
            return val
        prev = getattr(self, key) if hasattr(self, key) else None
        parsed = super().set_parameter_value(key, raw_value)
        if key in ("primary_turns", "secondary_turns") and parsed < 1.0:
            setattr(self, key, prev)
            raise ValueError("Windings must be >= 1 turn.")
        if key == "coupling_k" and parsed > 1.0:
            setattr(self, key, prev)
            raise ValueError("Coupling Factor (k) must be in (0, 1].")
        return parsed

    def _effective_params(self) -> tuple[float, float, float, float, float]:
        """Return `(r_p, r_s, l1, l2, m)` for Ideal or Non-Ideal mode."""
        n = self.turns_ratio
        if self.is_ideal:
            r_p = 1e-3
            r_s = 1e-3
            l1 = 10.0
            l2 = l1 * (n * n)
            k_eff = 1.0 - 1e-7
            m = k_eff * math.sqrt(l1 * l2)
        else:
            r_p = max(self.r_primary_ohm, 1e-6)
            r_s = max(self.r_secondary_ohm, 1e-6)
            l1 = max(self.l_primary_h, 1e-9)
            l2 = l1 * (n * n)
            k_eff = min(1.0 - 1e-7, max(1e-6, self.coupling_k))
            m = k_eff * math.sqrt(l1 * l2)
        return (r_p, r_s, l1, l2, m)

    def _companion_2port(
        self, dt: float, method: str
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return `(g11, g12, g22, ih_p, ih_s, r_p, r_s)` for a coupled transient step."""
        r_p, r_s, l1, l2, m = self._effective_params()
        if method == "backward_euler":
            alpha = dt
            beta = 0.0
        else:
            alpha = 0.5 * dt
            beta = 1.0

        inv_a = 1.0 / alpha
        z11 = r_p + l1 * inv_a
        z22 = r_s + l2 * inv_a
        z12 = m * inv_a
        det = z11 * z22 - z12 * z12
        g11 = z22 / det
        g22 = z11 / det
        g12 = -z12 / det

        vh_p = (l1 * self.i_p + m * self.i_s) * inv_a + beta * self.v_ind_p
        vh_s = (m * self.i_p + l2 * self.i_s) * inv_a + beta * self.v_ind_s
        ih_p = g11 * vh_p + g12 * vh_s
        ih_s = g12 * vh_p + g22 * vh_s
        return (g11, g12, g22, ih_p, ih_s, r_p, r_s)

    def companion_stamp(
        self,
        dt: float = 0.0,
        method: str = "trapezoidal",
        mode: str = "dc",
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        """Return symmetric 4x4 `(G_mat, rhs)` across `(p1, p2, s1, s2)`."""
        g_iso = 1e-12
        if mode == "dc" or dt <= 0.0:
            r_p, r_s, _, _, _ = self._effective_params()
            gp = 1.0 / r_p
            gs = 1.0 / r_s
            g_mat = (
                (gp, -gp, 0.0, 0.0),
                (-gp, gp + g_iso, 0.0, -g_iso),
                (0.0, 0.0, gs, -gs),
                (0.0, -g_iso, -gs, gs + g_iso),
            )
            return (g_mat, (0.0, 0.0, 0.0, 0.0))

        g11, g12, g22, ih_p, ih_s, _, _ = self._companion_2port(dt, method)
        g_mat = (
            (g11, -g11, g12, -g12),
            (-g11, g11 + g_iso, -g12, g12 - g_iso),
            (g12, -g12, g22, -g22),
            (-g12, g12 - g_iso, -g22, g22 + g_iso),
        )
        return (g_mat, (-ih_p, ih_p, -ih_s, ih_s))

    def update_companion_state(
        self,
        node_voltages: Mapping[str, float],
        *,
        dt: float = 0.0,
        method: str = "trapezoidal",
        mode: str = "dc",
    ) -> float:
        """Update primary/secondary currents and induced winding voltages after solve."""
        vp = float(node_voltages.get(self.node_p1, 0.0)) - float(
            node_voltages.get(self.node_p2, 0.0)
        )
        vs = float(node_voltages.get(self.node_s1, 0.0)) - float(
            node_voltages.get(self.node_s2, 0.0)
        )
        if mode == "dc" or dt <= 0.0:
            r_p, r_s, _, _, _ = self._effective_params()
            self.i_p = vp / r_p
            self.i_s = vs / r_s
            self.v_ind_p = 0.0
            self.v_ind_s = 0.0
        else:
            g11, g12, g22, ih_p, ih_s, r_p, r_s = self._companion_2port(dt, method)
            self.i_p = g11 * vp + g12 * vs + ih_p
            self.i_s = g12 * vp + g22 * vs + ih_s
            self.v_ind_p = vp - r_p * self.i_p
            self.v_ind_s = vs - r_s * self.i_s

        self.voltage = vs
        self.current = self.i_s
        self.power = abs(vp * self.i_p + vs * self.i_s)
        return self.i_s

    def ac_admittance_matrix(self, omega: float) -> tuple[tuple[complex, ...], ...]:
        """Return complex 4x4 small-signal admittance matrix at angular frequency `omega`."""
        r_p, r_s, l1, l2, m = self._effective_params()
        z11 = complex(r_p, omega * l1)
        z22 = complex(r_s, omega * l2)
        z12 = complex(0.0, omega * m)
        det = z11 * z22 - z12 * z12
        y11 = z22 / det
        y22 = z11 / det
        y12 = -z12 / det
        y_iso = complex(1e-12, 0.0)
        return (
            (y11, -y11, y12, -y12),
            (-y11, y11 + y_iso, -y12, y12 - y_iso),
            (y12, -y12, y22, -y22),
            (-y12, y12 - y_iso, -y22, y22 + y_iso),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_turns": self.primary_turns,
            "secondary_turns": self.secondary_turns,
            "turns_ratio": self.turns_ratio,
            "is_ideal": self.is_ideal,
            "l_primary_h": self.l_primary_h,
            "coupling_k": self.coupling_k,
            "r_primary_ohm": self.r_primary_ohm,
            "r_secondary_ohm": self.r_secondary_ohm,
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
            return "BLOWN (click to reset)"
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

    def set_parameter_value(self, key: str, raw_value: float | str) -> float:
        prev_rating = self.rating_amps
        parsed = super().set_parameter_value(key, raw_value)
        if key == "rating_amps":
            self.max_current = self.rating_amps
            if self.rating_amps > prev_rating:
                self.is_blown = False
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "rating_amps": self.rating_amps,
            "r_cold_ohm": self.r_cold_ohm,
            "r_blown_ohm": self.r_blown_ohm,
            "is_blown": self.is_blown,
        }

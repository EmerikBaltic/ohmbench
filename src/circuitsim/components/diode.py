"""Shockley diode component model with series resistance and reverse breakdown."""

from __future__ import annotations

import math
from typing import Any

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class Diode(Component):
    """Non-ideal Shockley PN-junction diode with ``N``, ``I_s``, ``R_s``, and ``V_br``.

    Terminal sign convention:
        - ``nodes[0]``: Anode (``node_anode``)
        - ``nodes[1]``: Cathode (``node_cathode``)
        - Positive terminal voltage ``v_ak = V_a - V_k`` drives forward current from
          anode to cathode obeying:
          ``v_ak = v_d + R_s * I_d(v_d)``
          with smooth forward exponential ``I_s * (exp(v_d / (N * V_T)) - 1)``
          and smooth reverse breakdown ``-I_bv * exp(-(v_d + V_br) / (N * V_T))``.
    """

    DEFAULT_IS: float = 2.52e-9
    DEFAULT_N: float = 1.752
    DEFAULT_RS: float = 0.568
    DEFAULT_BV: float = 100.0
    DEFAULT_IBV: float = 1e-4
    DEFAULT_VT: float = 0.0258563
    _EXP_MAX: float = 40.0

    def __init__(
        self,
        name: str,
        node_anode: str,
        node_cathode: str,
        *,
        is_sat: float | str = DEFAULT_IS,
        n: float | str = DEFAULT_N,
        rs: float | str = DEFAULT_RS,
        bv: float | str = DEFAULT_BV,
        ibv: float | str = DEFAULT_IBV,
        max_current: float | str = 0.2,
        rated_power: float | str = 0.5,
        thermal_voltage: float | str = DEFAULT_VT,
        model_name: str = "1N4148",
        label: str | None = None,
        show_value: bool = True,
        **spice_aliases: Any,
    ) -> None:
        raw_is = spice_aliases.pop("IS", spice_aliases.pop("i_s", is_sat))
        raw_n = spice_aliases.pop("N", spice_aliases.pop("n_ideality", n))
        raw_rs = spice_aliases.pop("RS", spice_aliases.pop("r_s", rs))
        raw_bv = spice_aliases.pop(
            "BV", spice_aliases.pop("v_br", spice_aliases.pop("max_reverse_v", bv))
        )
        raw_ibv = spice_aliases.pop("IBV", ibv)
        raw_vt = spice_aliases.pop("VT", spice_aliases.pop("vt", thermal_voltage))
        if spice_aliases:
            unknown = ", ".join(sorted(spice_aliases))
            raise TypeError(f"Diode {name!r} got unexpected keyword argument(s): {unknown}")

        val_is = parse_engineering_value(raw_is)
        val_n = parse_engineering_value(raw_n)
        val_rs = parse_engineering_value(raw_rs)
        val_bv = abs(parse_engineering_value(raw_bv))
        val_ibv = parse_engineering_value(raw_ibv)
        val_imax = parse_engineering_value(max_current)
        val_pmax = parse_engineering_value(rated_power)
        val_vt = parse_engineering_value(raw_vt)

        if val_is <= 0.0:
            raise ValueError(f"Diode {name!r}: is_sat must be > 0 A, got {val_is}.")
        if val_n <= 0.0:
            raise ValueError(f"Diode {name!r}: n must be > 0, got {val_n}.")
        if val_rs < 0.0:
            raise ValueError(f"Diode {name!r}: rs must be >= 0 ohm, got {val_rs}.")
        if val_bv <= 0.0:
            raise ValueError(f"Diode {name!r}: bv must be > 0 V, got {val_bv}.")
        if val_ibv <= 0.0:
            raise ValueError(f"Diode {name!r}: ibv must be > 0 A, got {val_ibv}.")
        if val_vt <= 0.0:
            raise ValueError(f"Diode {name!r}: thermal_voltage must be > 0 V, got {val_vt}.")

        super().__init__(
            name,
            (node_anode, node_cathode),
            label=label,
            show_value=show_value,
        )
        self.set_max_ratings(max_power=val_pmax, max_voltage=val_bv, max_current=val_imax)
        self.is_sat: float = val_is
        self.n: float = val_n
        self.rs: float = val_rs
        self.bv: float = val_bv
        self.ibv: float = val_ibv
        self.rated_power: float = val_pmax
        self.vt: float = val_vt
        self.model_name: str = str(model_name or "1N4148")
        self.g_min_leak: float = 1e-12
        self.operating_vd: float = 0.0
        self.operating_vak: float = 0.0

    @property
    def component_type(self) -> str:
        return "diode"

    @property
    def anode_node(self) -> str:
        return self.nodes[0]

    @property
    def cathode_node(self) -> str:
        return self.nodes[1]

    @property
    def v_br(self) -> float:
        return self.bv

    @v_br.setter
    def v_br(self, value: float) -> None:
        self.bv = abs(float(value))
        self.rated_voltage = self.bv

    @property
    def max_reverse_v(self) -> float:
        return self.bv

    @property
    def v_t_n(self) -> float:
        return self.n * self.vt

    @property
    def v_crit(self) -> float:
        """SPICE critical voltage for forward junction step limiting."""
        vtn = self.v_t_n
        ratio = max(vtn / (math.sqrt(2.0) * self.is_sat), math.e)
        return max(0.05, vtn * math.log(ratio))

    @property
    def primary_value(self) -> float:
        return self.bv

    @property
    def display_value(self) -> str:
        return self.model_name

    @property
    def conductance(self) -> float:
        """Small-signal local differential conductance around current operating point."""
        return self.companion_model()[0]

    def _forward_eval(self, vd: float) -> tuple[float, float]:
        vtn = self.v_t_n
        x = vd / vtn
        if x <= self._EXP_MAX:
            e = math.exp(x) if x > -80.0 else 0.0
            return self.is_sat * (e - 1.0), (self.is_sat / vtn) * e
        e_max = math.exp(self._EXP_MAX)
        i_f = self.is_sat * (e_max * (1.0 + (x - self._EXP_MAX)) - 1.0)
        g_f = (self.is_sat / vtn) * e_max
        return i_f, g_f

    def _reverse_eval(self, vd: float) -> tuple[float, float]:
        vtn = self.v_t_n
        x_r = -(vd + self.bv) / vtn
        if x_r <= -35.0:
            return 0.0, 0.0
        if x_r <= self._EXP_MAX:
            e = math.exp(x_r)
            return self.ibv * e, (self.ibv / vtn) * e
        e_max = math.exp(self._EXP_MAX)
        i_r = self.ibv * e_max * (1.0 + (x_r - self._EXP_MAX))
        g_r = (self.ibv / vtn) * e_max
        return i_r, g_r

    def junction_current(self, vd: float) -> float:
        """Internal PN-junction current `I_j(vd)` (without `rs` series drop)."""
        i_f, _ = self._forward_eval(vd)
        i_r, _ = self._reverse_eval(vd)
        return i_f - i_r + self.g_min_leak * vd

    def junction_conductance(self, vd: float) -> float:
        """Derivative `dI_j / dv_d` (strictly positive)."""
        _, g_f = self._forward_eval(vd)
        _, g_r = self._reverse_eval(vd)
        return g_f + g_r + self.g_min_leak

    def _junction_eval(self, vd: float) -> tuple[float, float]:
        i_f, g_f = self._forward_eval(vd)
        i_r, g_r = self._reverse_eval(vd)
        return (i_f - i_r + self.g_min_leak * vd), (g_f + g_r + self.g_min_leak)

    def terminal_model(self, v_ak: float, *, vd_guess: float | None = None) -> tuple[float, float, float]:
        """Solve exact ``v_ak = v_d + R_s * I_j(v_d)`` for ``(v_d, i_terminal, g_eq)``."""
        if self.rs <= 0.0:
            i_d, g_j = self._junction_eval(v_ak)
            return (v_ak, i_d, g_j)

        lo = min(0.0, v_ak)
        hi = max(0.0, v_ak)
        if lo == hi:
            i_d, g_j = self._junction_eval(lo)
            return (lo, i_d, g_j / (1.0 + self.rs * g_j))

        vd = lo if vd_guess is None else min(max(vd_guess, lo), hi)
        if vd == lo or vd == hi:
            vd = 0.5 * (lo + hi)

        for _ in range(40):
            i_j, g_j = self._junction_eval(vd)
            f_val = vd + self.rs * i_j - v_ak
            if abs(f_val) <= 1e-13 * max(1.0, abs(v_ak)):
                return (vd, i_j, g_j / (1.0 + self.rs * g_j))
            if f_val < 0.0:
                lo = max(lo, vd)
            else:
                hi = min(hi, vd)
            df = 1.0 + self.rs * g_j
            vd_next = vd - f_val / df
            if not (lo < vd_next < hi):
                vd_next = 0.5 * (lo + hi)
            if abs(vd_next - vd) <= 1e-13 * max(1.0, abs(vd)):
                i_j2, g_j2 = self._junction_eval(vd_next)
                return (vd_next, i_j2, g_j2 / (1.0 + self.rs * g_j2))
            vd = vd_next

        i_j, g_j = self._junction_eval(vd)
        return (vd, i_j, g_j / (1.0 + self.rs * g_j))

    def current_at_voltage(self, v_ak: float) -> float:
        """Return exact diode current (A) for terminal voltage `v_ak = V_a - V_k`."""
        return self.terminal_model(v_ak, vd_guess=self.operating_vd)[1]

    def limit_junction_voltage(self, v_d_new: float, v_d_old: float) -> tuple[float, bool]:
        """SPICE-style ``pnjlim`` logarithmic step limiter for Newton convergence."""
        vtn = self.v_t_n
        vcrit = self.v_crit
        dv = v_d_new - v_d_old
        if abs(dv) <= 2.0 * vtn:
            return (v_d_new, False)

        # Forward exponential region limiting (active well before overflow).
        if v_d_new > 0.4 * vcrit and dv > 2.0 * vtn:
            if v_d_old > 0.0:
                arg = 1.0 + dv / vtn
                if arg > 0.0:
                    v_lim = v_d_old + vtn * math.log(arg)
                else:
                    v_lim = vcrit
            else:
                v_lim = vtn * math.log(max(v_d_new / vtn, 1.0))
            return (min(v_d_new, v_lim), True)

        # Reverse breakdown knee limiting.
        if v_d_new < -0.5 * self.bv and dv < -2.0 * vtn:
            vr_new = -v_d_new - self.bv
            vr_old = -v_d_old - self.bv
            if vr_new > 0.0 and (vr_new - vr_old) > 2.0 * vtn:
                if vr_old > 0.0:
                    arg_r = 1.0 + (vr_new - vr_old) / vtn
                    vr_lim = vr_old + vtn * math.log(max(arg_r, 1.0))
                else:
                    vr_lim = vtn * math.log(max(vr_new / vtn, 1.0))
                return (-(self.bv + vr_lim), True)
            if v_d_old > -0.5 * self.bv and v_d_new < -self.bv - 2.0 * vtn:
                return (-self.bv - 0.5 * vtn, True)

        return (v_d_new, False)

    def companion_model(
        self, dt: float = 0.0, method: str = "trapezoidal"
    ) -> tuple[float, float]:
        """Return Norton tangent companion `(g_eq, i_eq)` around `operating_vd`.

        Satisfies `i_ak = g_eq * v_ak - i_eq` where `v_ak = v_d + R_s * I_j(v_d)`.
        """
        _ = (dt, method)
        i_op, g_j = self._junction_eval(self.operating_vd)
        v_ak_op = self.operating_vd + self.rs * i_op
        g_eq = g_j / (1.0 + self.rs * g_j)
        i_eq = g_eq * v_ak_op - i_op
        return (g_eq, i_eq)

    def reset_state(self) -> None:
        super().reset_state()
        self.operating_vd = 0.0
        self.operating_vak = 0.0

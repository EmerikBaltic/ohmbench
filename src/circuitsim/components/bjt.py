"""Bipolar Junction Transistor (NPN and PNP) Ebers-Moll / Gummel-Poon physics models (no Qt)."""

from __future__ import annotations

import math
from typing import Any

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class BJTBase(Component):
    """Base 3-terminal BJT (`B`, `C`, `E`) with Ebers-Moll transport + Early effect.

    Node ordering matches ``npn_bjt.json`` / ``pnp_bjt.json``:
        - ``nodes[0]``: ``B`` (Base)
        - ``nodes[1]``: ``C`` (Collector)
        - ``nodes[2]``: ``E`` (Emitter)
    """

    DEFAULT_BF: float = 100.0
    DEFAULT_BR: float = 1.0
    DEFAULT_IS: float = 1e-14
    DEFAULT_VAF: float = 100.0
    DEFAULT_VT: float = 0.0258563
    _EXP_MAX: float = 40.0
    _POLARITY: int = 1  # +1 for NPN, -1 for PNP

    def __init__(
        self,
        name: str,
        node_b: str,
        node_c: str,
        node_e: str,
        *,
        beta_f: float | str = DEFAULT_BF,
        beta_r: float | str = DEFAULT_BR,
        i_s: float | str = DEFAULT_IS,
        v_af: float | str = DEFAULT_VAF,
        max_power_w: float | str = 0.5,
        thermal_voltage: float | str = DEFAULT_VT,
        model_name: str = "",
        label: str | None = None,
        show_value: bool = True,
        **spice_aliases: Any,
    ) -> None:
        raw_bf = spice_aliases.pop("BF", spice_aliases.pop("bf", beta_f))
        raw_br = spice_aliases.pop("BR", spice_aliases.pop("br", beta_r))
        raw_is = spice_aliases.pop("IS", spice_aliases.pop("is_sat", i_s))
        raw_vaf = spice_aliases.pop("VAF", spice_aliases.pop("vaf", v_af))
        raw_pmax = spice_aliases.pop("max_power", max_power_w)
        raw_vt = spice_aliases.pop("VT", spice_aliases.pop("vt", thermal_voltage))
        if spice_aliases:
            unknown = ", ".join(sorted(spice_aliases))
            raise TypeError(f"BJT {name!r} got unexpected keyword argument(s): {unknown}")

        val_bf = parse_engineering_value(raw_bf)
        val_br = parse_engineering_value(raw_br)
        val_is = parse_engineering_value(raw_is)
        val_vaf = parse_engineering_value(raw_vaf)
        val_pmax = parse_engineering_value(raw_pmax)
        val_vt = parse_engineering_value(raw_vt)

        if val_bf <= 0.0:
            raise ValueError(f"BJT {name!r}: beta_f must be > 0 (got {val_bf}).")
        if val_br <= 0.0:
            raise ValueError(f"BJT {name!r}: beta_r must be > 0 (got {val_br}).")
        if val_is <= 0.0:
            raise ValueError(f"BJT {name!r}: i_s must be > 0 A (got {val_is}).")
        if val_vaf <= 0.0:
            raise ValueError(f"BJT {name!r}: v_af must be > 0 V (got {val_vaf}).")
        if val_pmax <= 0.0:
            raise ValueError(f"BJT {name!r}: max_power_w must be > 0 W (got {val_pmax}).")
        if val_vt <= 0.0:
            raise ValueError(f"BJT {name!r}: thermal_voltage must be > 0 V (got {val_vt}).")

        super().__init__(name, (node_b, node_c, node_e), label=label, show_value=show_value)
        self.beta_f: float = val_bf
        self.beta_r: float = val_br
        self.i_s: float = val_is
        self.v_af: float = val_vaf
        self.vt: float = val_vt
        self.model_name: str = model_name or ("2N3904" if self._POLARITY > 0 else "2N3906")
        self.set_max_ratings(max_power=val_pmax, max_voltage=40.0, max_current=0.5)
        self.g_min_leak: float = 1e-12

        self.operating_vbe: float = 0.0
        self.operating_vbc: float = 0.0
        self.i_b: float = 0.0
        self.i_c: float = 0.0
        self.i_e: float = 0.0

    @property
    def node_b(self) -> str:
        return self.nodes[0]

    @property
    def node_c(self) -> str:
        return self.nodes[1]

    @property
    def node_e(self) -> str:
        return self.nodes[2]

    @property
    def max_power_w(self) -> float:
        return self.max_power

    @max_power_w.setter
    def max_power_w(self, value: float) -> None:
        self.set_max_ratings(max_power=float(value))

    @property
    def BF(self) -> float:
        return self.beta_f

    @BF.setter
    def BF(self, value: float) -> None:
        self.beta_f = float(value)

    @property
    def BR(self) -> float:
        return self.beta_r

    @BR.setter
    def BR(self, value: float) -> None:
        self.beta_r = float(value)

    @property
    def IS(self) -> float:
        return self.i_s

    @IS.setter
    def IS(self, value: float) -> None:
        self.i_s = float(value)

    @property
    def VAF(self) -> float:
        return self.v_af

    @VAF.setter
    def VAF(self, value: float) -> None:
        self.v_af = float(value)

    @property
    def display_value(self) -> str:
        return f"β={self.beta_f:g}"

    @property
    def v_crit(self) -> float:
        ratio = max(self.vt / (math.sqrt(2.0) * self.i_s), math.e)
        return max(0.05, self.vt * math.log(ratio))

    def _diode_eval(self, v_j: float) -> tuple[float, float]:
        x = v_j / self.vt
        if x <= self._EXP_MAX:
            e = math.exp(x) if x > -80.0 else 0.0
            return self.i_s * (e - 1.0), (self.i_s / self.vt) * e
        e_max = math.exp(self._EXP_MAX)
        i_val = self.i_s * (e_max * (1.0 + (x - self._EXP_MAX)) - 1.0)
        g_val = (self.i_s / self.vt) * e_max
        return i_val, g_val

    def limit_junction_voltage(self, v_new: float, v_old: float) -> tuple[float, bool]:
        """SPICE-style ``pnjlim`` logarithmic step limiter for BJT BE/BC junctions."""
        vt = self.vt
        vcrit = self.v_crit
        dv = v_new - v_old
        if abs(dv) <= 2.0 * vt:
            return (v_new, False)
        if v_new > 0.4 * vcrit and dv > 2.0 * vt:
            if v_old > 0.0:
                arg = 1.0 + dv / vt
                v_lim = v_old + vt * math.log(arg) if arg > 0.0 else vcrit
            else:
                v_lim = vt * math.log(max(v_new / vt, 1.0))
            return (min(v_new, v_lim), True)
        return (v_new, False)

    def _intrinsic_eval(
        self, v_be: float, v_bc: float
    ) -> tuple[float, float, float, float, float, float, float]:
        """Evaluate intrinsic Ebers-Moll currents `(ib, ic, ie)` and partials w.r.t. `(vbe, vbc)`."""
        i_f, g_f = self._diode_eval(v_be)
        i_r, g_r = self._diode_eval(v_bc)
        v_ce = v_be - v_bc
        if self.v_af > 0.0:
            raw_early = 1.0 + v_ce / self.v_af
            if raw_early >= 0.2:
                early = raw_early
                d_early_dvce = 1.0 / self.v_af
            else:
                early = 0.2
                d_early_dvce = 0.0
        else:
            early = 1.0
            d_early_dvce = 0.0

        ib = i_f / self.beta_f + i_r / self.beta_r + self.g_min_leak * (v_be + v_bc)
        ic = (i_f - i_r) * early - i_r / self.beta_r + self.g_min_leak * (v_be - v_bc)
        ie = -(ib + ic)

        dib_dvbe = g_f / self.beta_f + self.g_min_leak
        dib_dvbc = g_r / self.beta_r + self.g_min_leak
        dic_dvbe = g_f * early + (i_f - i_r) * d_early_dvce + self.g_min_leak
        dic_dvbc = -g_r * early - (i_f - i_r) * d_early_dvce - g_r / self.beta_r - self.g_min_leak
        return (ib, ic, ie, dib_dvbe, dib_dvbc, dic_dvbe, dic_dvbc)

    def terminal_currents_at(
        self, v_b: float, v_c: float, v_e: float
    ) -> tuple[float, float, float]:
        """Return actual currents `(I_B, I_C, I_E)` flowing into `(B, C, E)` terminals."""
        s = float(self._POLARITY)
        v_be = s * (v_b - v_e)
        v_bc = s * (v_b - v_c)
        ib, ic, ie, *_ = self._intrinsic_eval(v_be, v_bc)
        return (s * ib, s * ic, s * ie)

    def companion_stamp(
        self,
        v_be: float | None = None,
        v_bc: float | None = None,
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
        """Return 3x3 Jacobian conductance matrix and 3-element RHS Norton vector for `(B, C, E)`."""
        s = float(self._POLARITY)
        vbe0 = self.operating_vbe if v_be is None else float(v_be)
        vbc0 = self.operating_vbc if v_bc is None else float(v_bc)
        ib0, ic0, ie0, dib_dvbe, dib_dvbc, dic_dvbe, dic_dvbc = self._intrinsic_eval(vbe0, vbc0)
        die_dvbe = -(dib_dvbe + dic_dvbe)
        die_dvbc = -(dib_dvbc + dic_dvbc)

        # Chain rule from v_be = s*(V_B - V_E), v_bc = s*(V_B - V_C): s*s = 1
        g_mat = (
            (dib_dvbe + dib_dvbc, -dib_dvbc, -dib_dvbe),
            (dic_dvbe + dic_dvbc, -dic_dvbc, -dic_dvbe),
            (die_dvbe + die_dvbc, -die_dvbc, -die_dvbe),
        )
        # RHS offset: KCL sum = 0 -> G*V = G*V0 - I0
        rhs_b = s * (dib_dvbe * vbe0 + dib_dvbc * vbc0 - ib0)
        rhs_c = s * (dic_dvbe * vbe0 + dic_dvbc * vbc0 - ic0)
        rhs_e = s * (die_dvbe * vbe0 + die_dvbc * vbc0 - ie0)
        return (g_mat, (rhs_b, rhs_c, rhs_e))

    def reset_state(self) -> None:
        super().reset_state()
        self.operating_vbe = 0.0
        self.operating_vbc = 0.0
        self.i_b = 0.0
        self.i_c = 0.0
        self.i_e = 0.0


class NPNBJT(BJTBase):
    """NPN Bipolar Junction Transistor (`B`, `C`, `E`)."""

    _POLARITY = 1

    @property
    def component_type(self) -> str:
        return "npn_bjt"


class PNPBJT(BJTBase):
    """PNP Bipolar Junction Transistor (`B`, `C`, `E`)."""

    _POLARITY = -1

    @property
    def component_type(self) -> str:
        return "pnp_bjt"

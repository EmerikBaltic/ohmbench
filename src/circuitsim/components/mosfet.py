"""Level-1 Shichman-Hodges MOSFET (NMOS and PMOS) physics models (no Qt)."""

from __future__ import annotations

from typing import Any

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class MOSFETBase(Component):
    """Base 3-terminal Level-1 MOSFET (`G`, `D`, `S`) with Cutoff, Triode, and Saturation.

    Node ordering matches ``nmos.json`` / ``pmos.json``:
        - ``nodes[0]``: ``G`` (Gate)
        - ``nodes[1]``: ``D`` (Drain)
        - ``nodes[2]``: ``S`` (Source)
    """

    DEFAULT_VTH: float = 1.5
    DEFAULT_KP: float = 0.02
    DEFAULT_LAMBDA: float = 0.01
    _POLARITY: int = 1  # +1 for NMOS, -1 for PMOS

    def __init__(
        self,
        name: str,
        node_g: str,
        node_d: str,
        node_s: str,
        *,
        v_th: float | str | None = None,
        k_p: float | str = DEFAULT_KP,
        lambda_: float | str = DEFAULT_LAMBDA,
        max_power_w: float | str = 0.5,
        model_name: str = "",
        label: str | None = None,
        show_value: bool = True,
        **spice_aliases: Any,
    ) -> None:
        default_vth = self.DEFAULT_VTH if self._POLARITY > 0 else -self.DEFAULT_VTH
        raw_vth = spice_aliases.pop(
            "VTO", spice_aliases.pop("vto", default_vth if v_th is None else v_th)
        )
        raw_kp = spice_aliases.pop("KP", spice_aliases.pop("kp", k_p))
        raw_lam = spice_aliases.pop(
            "LAMBDA", spice_aliases.pop("lambda_val", lambda_)
        )
        raw_pmax = spice_aliases.pop("max_power", max_power_w)
        if spice_aliases:
            unknown = ", ".join(sorted(spice_aliases))
            raise TypeError(f"MOSFET {name!r} got unexpected keyword argument(s): {unknown}")

        val_vth = parse_engineering_value(raw_vth)
        val_kp = parse_engineering_value(raw_kp)
        val_lam = parse_engineering_value(raw_lam)
        val_pmax = parse_engineering_value(raw_pmax)

        if val_kp <= 0.0:
            raise ValueError(f"MOSFET {name!r}: k_p must be > 0 A/V^2 (got {val_kp}).")
        if val_lam < 0.0:
            raise ValueError(f"MOSFET {name!r}: lambda_ must be >= 0 1/V (got {val_lam}).")
        if val_pmax <= 0.0:
            raise ValueError(f"MOSFET {name!r}: max_power_w must be > 0 W (got {val_pmax}).")

        super().__init__(name, (node_g, node_d, node_s), label=label, show_value=show_value)
        self.v_th: float = val_vth
        self.k_p: float = val_kp
        self.lambda_: float = val_lam
        self.model_name: str = model_name or ("2N7000" if self._POLARITY > 0 else "BSS84")
        self.set_max_ratings(max_power=val_pmax, max_voltage=60.0, max_current=1.0)
        self.g_min_leak: float = 1e-12

        self.operating_vg: float = 0.0
        self.operating_vd: float = 0.0
        self.operating_vs: float = 0.0
        self.i_d: float = 0.0

    @property
    def node_g(self) -> str:
        return self.nodes[0]

    @property
    def node_d(self) -> str:
        return self.nodes[1]

    @property
    def node_s(self) -> str:
        return self.nodes[2]

    @property
    def max_power_w(self) -> float:
        return self.max_power

    @max_power_w.setter
    def max_power_w(self, value: float) -> None:
        self.set_max_ratings(max_power=float(value))

    @property
    def VTO(self) -> float:
        return self.v_th

    @VTO.setter
    def VTO(self, value: float) -> None:
        self.v_th = float(value)

    @property
    def KP(self) -> float:
        return self.k_p

    @KP.setter
    def KP(self, value: float) -> None:
        self.k_p = float(value)

    @property
    def LAMBDA(self) -> float:
        return self.lambda_

    @LAMBDA.setter
    def LAMBDA(self, value: float) -> None:
        self.lambda_ = float(value)

    @property
    def display_value(self) -> str:
        return f"Vth={self.v_th:g}V"

    def _intrinsic_eval(
        self, v_g: float, v_d: float, v_s: float
    ) -> tuple[float, float, float, float]:
        """Return `(i_d, dg, dd, ds)` for drain current INTO node `D`."""
        s = float(self._POLARITY)
        v_t_eff = s * self.v_th
        v_ds_raw = s * (v_d - v_s)

        if v_ds_raw >= 0.0:
            mode = 1.0
            v_gs = s * (v_g - v_s)
            v_ds = v_ds_raw
        else:
            mode = -1.0
            v_gs = s * (v_g - v_d)
            v_ds = -v_ds_raw

        v_ov = v_gs - v_t_eff
        mod = 1.0 + self.lambda_ * v_ds
        if v_ov <= 0.0:
            i_ds = 0.0
            g_m = 0.0
            g_ds = self.g_min_leak
        elif v_ds < v_ov:
            i_core = self.k_p * (v_ov * v_ds - 0.5 * v_ds * v_ds)
            i_ds = i_core * mod
            g_m = self.k_p * v_ds * mod
            g_ds = self.k_p * (v_ov - v_ds) * mod + i_core * self.lambda_ + self.g_min_leak
        else:
            i_core = 0.5 * self.k_p * v_ov * v_ov
            i_ds = i_core * mod
            g_m = self.k_p * v_ov * mod
            g_ds = i_core * self.lambda_ + self.g_min_leak

        i_d = s * mode * i_ds + self.g_min_leak * (v_d - v_s)
        if mode > 0.0:
            d_vg = g_m
            d_vd = g_ds + self.g_min_leak
            d_vs = -(g_m + g_ds + self.g_min_leak)
        else:
            d_vg = -g_m
            d_vd = g_m + g_ds + self.g_min_leak
            d_vs = -(g_ds + self.g_min_leak)
        return (i_d, d_vg, d_vd, d_vs)

    def drain_current_at(self, v_g: float, v_d: float, v_s: float) -> float:
        """Return drain current `I_D` flowing into `D` (A)."""
        return self._intrinsic_eval(v_g, v_d, v_s)[0]

    def limit_terminal_voltages(
        self, v_g_new: float, v_d_new: float, v_s_new: float
    ) -> tuple[float, float, float, bool]:
        """Step-limit gate/drain/source voltages for smooth Newton convergence."""
        max_step = 2.0
        limited = False
        out: list[float] = []
        for vn, vo in (
            (v_g_new, self.operating_vg),
            (v_d_new, self.operating_vd),
            (v_s_new, self.operating_vs),
        ):
            dv = vn - vo
            if abs(dv) > max_step:
                out.append(vo + (max_step if dv > 0.0 else -max_step))
                limited = True
            else:
                out.append(vn)
        return (out[0], out[1], out[2], limited)

    def companion_stamp(
        self,
        v_g: float | None = None,
        v_d: float | None = None,
        v_s: float | None = None,
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
        """Return 3x3 Jacobian matrix and 3-element Norton RHS vector for `(G, D, S)`."""
        vg0 = self.operating_vg if v_g is None else float(v_g)
        vd0 = self.operating_vd if v_d is None else float(v_d)
        vs0 = self.operating_vs if v_s is None else float(v_s)

        i_d0, d_vg, d_vd, d_vs = self._intrinsic_eval(vg0, vd0, vs0)
        g_mat = (
            (0.0, 0.0, 0.0),
            (d_vg, d_vd, d_vs),
            (-d_vg, -d_vd, -d_vs),
        )
        rhs_d = d_vg * vg0 + d_vd * vd0 + d_vs * vs0 - i_d0
        return (g_mat, (0.0, rhs_d, -rhs_d))

    def reset_state(self) -> None:
        super().reset_state()
        self.operating_vg = 0.0
        self.operating_vd = 0.0
        self.operating_vs = 0.0
        self.i_d = 0.0


class NMOS(MOSFETBase):
    """N-Channel Enhancement MOSFET (`G`, `D`, `S`)."""

    _POLARITY = 1

    @property
    def component_type(self) -> str:
        return "nmos"


class PMOS(MOSFETBase):
    """P-Channel Enhancement MOSFET (`G`, `D`, `S`)."""

    _POLARITY = -1

    @property
    def component_type(self) -> str:
        return "pmos"

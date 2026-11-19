"""Four-terminal Benchtop DSO-1062D Oscilloscope probe component (Qt-free)."""

from __future__ import annotations

from typing import Mapping

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class Oscilloscope(Component):
    """Dual-channel benchtop oscilloscope probe (`CH1`, `CH2`, `EXT_TRIG`, `GND`).

    Each input pin (`CH1`, `CH2`, `EXT_TRIG`) presents a high input impedance
    `r_in` (default ``10 MOhm``) referenced to `GND` when `GND` is wired in the
    circuit, or cleanly referenced to circuit ground `'0'` when `GND` is left
    unwired on the schematic so floating-node errors never occur.
    """

    PIN_NAMES: tuple[str, ...] = ("CH1", "CH2", "EXT_TRIG", "GND")

    def __init__(
        self,
        name: str,
        node_ch1: str | None = None,
        node_ch2: str | None = None,
        node_ext_trig: str | None = None,
        node_gnd: str | None = None,
        *,
        r_in: float | str = 10e6,
        dso_state_json: str = "",
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        resolved = (
            node_ch1 or f"{name}_CH1",
            node_ch2 or f"{name}_CH2",
            node_ext_trig or f"{name}_EXT_TRIG",
            node_gnd or f"{name}_GND",
        )
        super().__init__(
            name=name,
            nodes=resolved,
            label=label,
            show_value=show_value,
            color=color,
        )
        parsed_r = parse_engineering_value(r_in)
        if parsed_r <= 0.0:
            raise ValueError(f"Oscilloscope {name!r}: r_in must be > 0 Ω, got {parsed_r}.")
        self.r_in: float = parsed_r
        self.dso_state_json: str = str(dso_state_json or "")

        # Live telemetry (differential w.r.t. GND)
        self.v_ch1: float = 0.0
        self.v_ch2: float = 0.0
        self.v_ext: float = 0.0
        self.v_gnd: float = 0.0
        self.mini_preview_ch1: tuple[float, ...] = ()
        self.mini_preview_ch2: tuple[float, ...] = ()

    @property
    def component_type(self) -> str:
        return "oscilloscope"

    @property
    def pin_names(self) -> tuple[str, ...]:
        return self.PIN_NAMES

    @property
    def node_ch1(self) -> str:
        return self.nodes[0]

    @property
    def node_ch2(self) -> str:
        return self.nodes[1]

    @property
    def node_ext_trig(self) -> str:
        return self.nodes[2]

    @property
    def node_gnd(self) -> str:
        return self.nodes[3]

    @property
    def display_value(self) -> str:
        return "DSO-1062D"

    def companion_stamp(
        self, *, gnd_wired: bool = True
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        """Return 4x4 MNA admittance matrix and RHS vector for (`CH1`, `CH2`, `EXT_TRIG`, `GND`)."""
        g = 1.0 / self.r_in
        if gnd_wired:
            mat = (
                (g, 0.0, 0.0, -g),
                (0.0, g, 0.0, -g),
                (0.0, 0.0, g, -g),
                (-g, -g, -g, 3.0 * g + 1e-12),
            )
        else:
            mat = (
                (g, 0.0, 0.0, 0.0),
                (0.0, g, 0.0, 0.0),
                (0.0, 0.0, g, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        return mat, (0.0, 0.0, 0.0, 0.0)

    def update_telemetry(
        self, node_voltages: Mapping[str, float], *, gnd_wired: bool = True
    ) -> float:
        """Update probe terminal voltages relative to `GND` (or ground if unwired)."""
        vg = float(node_voltages.get(self.node_gnd, 0.0)) if gnd_wired else 0.0
        self.v_gnd = vg
        self.v_ch1 = float(node_voltages.get(self.node_ch1, 0.0)) - vg
        self.v_ch2 = float(node_voltages.get(self.node_ch2, 0.0)) - vg
        self.v_ext = float(node_voltages.get(self.node_ext_trig, 0.0)) - vg
        self.voltage = self.v_ch1
        self.current = self.v_ch1 / self.r_in
        self.power = abs(self.v_ch1 * self.current)
        return self.current

    def reset_state(self) -> None:
        """Clear electrical readings and miniature screen preview buffers."""
        super().reset_state()
        self.v_ch1 = 0.0
        self.v_ch2 = 0.0
        self.v_ext = 0.0
        self.v_gnd = 0.0
        self.mini_preview_ch1 = ()
        self.mini_preview_ch2 = ()

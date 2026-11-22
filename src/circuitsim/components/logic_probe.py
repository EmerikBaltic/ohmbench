"""Single-pin digital Logic Probe component (Qt-free)."""

from __future__ import annotations

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value

LOGIC_PROBE_FAMILIES: dict[str, tuple[float, float]] = {
    "5V_CMOS": (1.5, 3.5),
    "5V_TTL": (0.8, 2.0),
    "3V3_LVCMOS": (0.8, 2.0),
}


class LogicProbe(Component):
    """1-pin digital logic probe sensing HIGH (`H`), LOW (`L`), or UNDEFINED (`?`).

    Stamps a `10 MOhm` (`r_in`) conductance from its input terminal `in` to
    ground `'0'` so unloaded nodes stay well-conditioned without disturbing logic levels.
    """

    PIN_NAMES: tuple[str, ...] = ("in",)

    def __init__(
        self,
        name: str,
        in_node: str | None = None,
        *,
        family: str = "5V_CMOS",
        v_il: float | None = None,
        v_ih: float | None = None,
        r_in: float | str = 10e6,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        resolved_in = in_node or f"{name}_in"
        super().__init__(
            name=name,
            nodes=(resolved_in,),
            label=label,
            show_value=show_value,
            color=color,
        )
        parsed_r = parse_engineering_value(r_in)
        if parsed_r <= 0.0:
            raise ValueError(f"LogicProbe {name!r}: r_in must be > 0 Ω.")
        norm_fam = str(family).strip() if str(family).strip() in LOGIC_PROBE_FAMILIES else "5V_CMOS"
        default_vil, default_vih = LOGIC_PROBE_FAMILIES[norm_fam]
        self.family: str = norm_fam
        self.v_il: float = float(default_vil if v_il is None else parse_engineering_value(v_il))
        self.v_ih: float = float(default_vih if v_ih is None else parse_engineering_value(v_ih))
        if self.v_il > self.v_ih:
            self.v_ih = self.v_il + 0.1
        self.r_in: float = parsed_r

        self.last_voltage: float = 0.0
        self.logic_state: str = "L"

    @property
    def component_type(self) -> str:
        return "logic_probe"

    @property
    def conductance(self) -> float:
        return 1.0 / self.r_in

    @property
    def display_value(self) -> str:
        return f"{self.logic_state} ({self.last_voltage:.2f} V)"

    def evaluate_voltage(self, voltage: float) -> str:
        """Update voltage/current telemetry and return `'H'`, `'L'`, or `'?'`."""
        v = float(voltage)
        self.last_voltage = v
        self.voltage = v
        self.current = v / self.r_in
        self.power = v * self.current
        if v >= self.v_ih:
            state = "H"
        elif v <= self.v_il:
            state = "L"
        else:
            state = "?"
        self.logic_state = state
        return state

    def set_family_preset(self, family: str) -> None:
        key = str(family).strip()
        if key in LOGIC_PROBE_FAMILIES:
            vil, vih = LOGIC_PROBE_FAMILIES[key]
            self.family = key
            self.v_il = float(vil)
            self.v_ih = float(vih)
        else:
            self.family = key or "Custom"
        self.evaluate_voltage(self.last_voltage)

    def set_parameter_value(self, key: str, raw_value: float | str) -> float:
        if key == "family":
            self.set_family_preset(str(raw_value))
            return 0.0
        val = super().set_parameter_value(key, raw_value)
        self.evaluate_voltage(self.last_voltage)
        return val

    def reset_state(self) -> None:
        super().reset_state()
        self.last_voltage = 0.0
        self.logic_state = "L"

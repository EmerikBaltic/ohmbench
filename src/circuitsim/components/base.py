"""Base component abstractions for CircuitSim (pure Python, no Qt imports)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from circuitsim.dialogs.value_parser import (
    format_engineering_value,
    parse_engineering_value,
)


@dataclass(frozen=True)
class ComponentParameter:
    """Descriptor for an editable numerical parameter of a `Component`.

    Used by both the Step 2C quick value dialog and the Milestone 3 full
    properties dialog, as well as headless live parameter updates.
    """

    key: str
    label: str
    unit: str
    allow_zero: bool = False
    allow_negative: bool = False
    live_editable: bool = True
    choices: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""


_DEFAULT_PARAMETERS_BY_TYPE: dict[str, tuple[ComponentParameter, ...]] = {
    "resistor": (
        ComponentParameter("resistance", "Resistance", "Ω"),
    ),
    "capacitor": (
        ComponentParameter("capacitance", "Capacitance", "F"),
        ComponentParameter(
            "initial_voltage",
            "Initial Voltage",
            "V",
            allow_zero=True,
            allow_negative=True,
            live_editable=False,
        ),
        ComponentParameter("esr", "Series ESR", "Ω", allow_zero=True),
    ),
    "inductor": (
        ComponentParameter("inductance", "Inductance", "H"),
        ComponentParameter(
            "dcr", "Winding Resistance (DCR)", "Ω", allow_zero=True
        ),
    ),
    "dc_voltage": (
        ComponentParameter(
            "dc_voltage", "Voltage", "V", allow_zero=True, allow_negative=True
        ),
        ComponentParameter(
            "r_internal", "Internal Resistance", "Ω", allow_zero=True
        ),
    ),
    "ac_voltage": (
        ComponentParameter("amplitude", "Amplitude (Peak)", "V", allow_zero=True),
        ComponentParameter("frequency", "Frequency", "Hz"),
        ComponentParameter(
            "dc_offset",
            "DC Offset",
            "V",
            allow_zero=True,
            allow_negative=True,
        ),
    ),
    "wav_voltage": (
        ComponentParameter("gain_v", "Gain", "V/FS", allow_negative=True, allow_zero=True),
        ComponentParameter(
            "dc_offset",
            "DC Offset",
            "V",
            allow_zero=True,
            allow_negative=True,
        ),
        ComponentParameter("r_internal", "Internal Resistance", "Ω", allow_zero=True),
    ),
    "dc_current": (
        ComponentParameter(
            "dc_current", "Current", "A", allow_zero=True, allow_negative=True
        ),
    ),
    "lightbulb": (
        ComponentParameter("rated_power", "Rated Power", "W"),
        ComponentParameter("rated_voltage", "Rated Voltage", "V"),
    ),
    "led": (
        ComponentParameter("forward_voltage", "Forward Voltage", "V"),
        ComponentParameter("rated_current", "Rated Current", "A"),
        ComponentParameter("max_current", "Max Current", "A"),
    ),
    "diode": (
        ComponentParameter("is_sat", "Saturation Current (Is)", "A"),
        ComponentParameter("n", "Ideality Factor (N)", ""),
        ComponentParameter("rs", "Series Resistance (Rs)", "Ω", allow_zero=True),
        ComponentParameter("bv", "Breakdown Voltage (Vbr)", "V"),
        ComponentParameter("max_current", "Max Forward Current", "A"),
        ComponentParameter("max_power", "Max Power Rating", "W"),
    ),
    "zener_diode": (
        ComponentParameter("v_z", "Zener Voltage (Vz / BV)", "V"),
        ComponentParameter("i_s", "Saturation Current (Is)", "A"),
        ComponentParameter("n", "Ideality Factor (N)", ""),
        ComponentParameter("r_s", "Series Resistance (Rs)", "Ω", allow_zero=True),
        ComponentParameter("max_power_w", "Max Power Rating", "W"),
    ),
    "npn_bjt": (
        ComponentParameter("beta_f", "Forward Beta (BF)", ""),
        ComponentParameter("beta_r", "Reverse Beta (BR)", ""),
        ComponentParameter("i_s", "Saturation Current (IS)", "A"),
        ComponentParameter("v_af", "Early Voltage (VAF)", "V"),
        ComponentParameter("max_power_w", "Max Power Rating", "W"),
    ),
    "pnp_bjt": (
        ComponentParameter("beta_f", "Forward Beta (BF)", ""),
        ComponentParameter("beta_r", "Reverse Beta (BR)", ""),
        ComponentParameter("i_s", "Saturation Current (IS)", "A"),
        ComponentParameter("v_af", "Early Voltage (VAF)", "V"),
        ComponentParameter("max_power_w", "Max Power Rating", "W"),
    ),
    "nmos": (
        ComponentParameter("v_th", "Threshold Voltage (VTO)", "V", allow_zero=True, allow_negative=True),
        ComponentParameter("k_p", "Transconductance (KP)", "A/V²"),
        ComponentParameter("lambda_", "Channel Modulation (LAMBDA)", "1/V", allow_zero=True),
        ComponentParameter("max_power_w", "Max Power Rating", "W"),
    ),
    "pmos": (
        ComponentParameter("v_th", "Threshold Voltage (VTO)", "V", allow_zero=True, allow_negative=True),
        ComponentParameter("k_p", "Transconductance (KP)", "A/V²"),
        ComponentParameter("lambda_", "Channel Modulation (LAMBDA)", "1/V", allow_zero=True),
        ComponentParameter("max_power_w", "Max Power Rating", "W"),
    ),
    "opamp": (
        ComponentParameter("a_ol", "Open-Loop Gain (Aol)", ""),
        ComponentParameter("r_in", "Input Resistance", "Ω"),
        ComponentParameter("r_out", "Output Resistance", "Ω"),
        ComponentParameter("v_rail_pos", "Positive Rail (+Vcc)", "V", allow_zero=True, allow_negative=True),
        ComponentParameter("v_rail_neg", "Negative Rail (-Vee)", "V", allow_zero=True, allow_negative=True),
    ),
    "comparator": (
        ComponentParameter("v_high", "Output High (Voh)", "V", allow_zero=True, allow_negative=True),
        ComponentParameter("v_low", "Output Low (Vol)", "V", allow_zero=True, allow_negative=True),
        ComponentParameter("v_offset", "Input Offset (Vos)", "V", allow_zero=True, allow_negative=True),
        ComponentParameter("a_ol", "Voltage Gain (Aol)", ""),
        ComponentParameter("r_in", "Input Resistance", "Ω"),
        ComponentParameter("r_out", "Output Resistance", "Ω"),
    ),
    "ne555": (
        ComponentParameter("r_div", "Divider Ladder R (5k)", "Ω"),
        ComponentParameter("r_out", "Output Resistance", "Ω"),
        ComponentParameter("r_dis_on", "Discharge ON Resistance", "Ω"),
        ComponentParameter("v_reset_th", "Reset Threshold", "V"),
    ),
    "switch": (
        ComponentParameter("r_on", "ON Resistance", "Ω"),
        ComponentParameter("v_breakdown", "Breakdown Voltage", "V"),
    ),
    "transformer": (
        ComponentParameter("turns_ratio", "Turns Ratio (Ns/Np)", ""),
        ComponentParameter("r_winding_ohm", "Winding Resistance", "Ω"),
        ComponentParameter("l_primary_h", "Primary Inductance", "H"),
        ComponentParameter("coupling_k", "Coupling Factor (k)", ""),
    ),
    "spdt_relay": (
        ComponentParameter("coil_resistance_ohm", "Coil Resistance", "Ω"),
        ComponentParameter("pull_in_voltage_v", "Pull-In Voltage", "V"),
        ComponentParameter("drop_out_voltage_v", "Drop-Out Voltage", "V", allow_zero=True),
    ),
    "fuse": (
        ComponentParameter("rating_amps", "Current Rating", "A"),
        ComponentParameter("r_cold_ohm", "Cold Resistance", "Ω"),
    ),
    "voltmeter": (
        ComponentParameter("r_internal", "Input Resistance", "Ω"),
    ),
    "mcu": (
        ComponentParameter("v_cc", "Logic / ADC Supply (Vcc)", "V"),
        ComponentParameter("r_adc_in", "ADC Input Impedance", "Ω"),
        ComponentParameter("r_dig_out", "GPIO Output Impedance", "Ω"),
    ),
    "ammeter": (
        ComponentParameter("r_internal", "Series Resistance", "Ω"),
    ),
    "ground": (),
}

_LOGIC_GATE_PARAMETERS: tuple[ComponentParameter, ...] = (
    ComponentParameter("t_pd", "Propagation Delay", "s", allow_zero=True),
    ComponentParameter("t_edge", "Rise/Fall Time", "s", allow_zero=True),
    ComponentParameter("v_il", "Input Low Max", "V", allow_zero=True, allow_negative=True),
    ComponentParameter("v_ih", "Input High Min", "V", allow_zero=True, allow_negative=True),
    ComponentParameter("v_ol", "Output Low", "V", allow_zero=True, allow_negative=True),
    ComponentParameter("v_oh", "Output High", "V", allow_zero=True, allow_negative=True),
    ComponentParameter("r_in", "Input Resistance", "Ω"),
    ComponentParameter("r_out", "Output Resistance", "Ω"),
)

for _gate_op in ("not", "and", "or", "nand", "nor", "xor", "xnor"):
    _DEFAULT_PARAMETERS_BY_TYPE[f"gate_{_gate_op}"] = _LOGIC_GATE_PARAMETERS

_SEQUENTIAL_PARAMETERS: tuple[ComponentParameter, ...] = (
    ComponentParameter("v_high", "Output High (Voh)", "V", allow_zero=True, allow_negative=True),
    ComponentParameter("v_low", "Output Low (Vol)", "V", allow_zero=True, allow_negative=True),
    ComponentParameter("v_threshold", "Logic Threshold (Vth)", "V", allow_zero=True),
    ComponentParameter("r_in", "Input Resistance", "Ω"),
    ComponentParameter("r_out", "Output Resistance", "Ω"),
)

for _seq_t in ("d_flipflop", "jk_flipflop", "counter_4bit"):
    _DEFAULT_PARAMETERS_BY_TYPE[_seq_t] = _SEQUENTIAL_PARAMETERS

_DEFAULT_PARAMETERS_BY_TYPE["seven_seg_display"] = (
    ComponentParameter("v_threshold", "Input Threshold (Vth)", "V", allow_zero=True),
    ComponentParameter("r_in", "Input Resistance", "Ω"),
)


class Component(ABC):
    """Abstract base class for all electrical components in a circuit.

    Attributes:
        name: Unique identifier in the circuit (e.g., "R1", "V1").
        nodes: Tuple of node names to which the component terminals connect.
        label: User-visible display label (defaults to `name`).
        show_value: Whether the component's value should be shown on the canvas.
        color: Optional display color string (e.g. hex code) for the UI.
        voltage: Latest solved voltage across the component (V_pos - V_neg), in Volts.
        current: Latest solved current through the component (from pos to neg), in Amperes.
        power: Latest solved power dissipated/absorbed by the component (V * I), in Watts.
    """

    def __init__(
        self,
        name: str,
        nodes: tuple[str, ...],
        *,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        if not name or not name.strip():
            raise ValueError("Component name must be a non-empty string.")
        self.name: str = name.strip()
        self.nodes: tuple[str, ...] = tuple(str(n).strip() for n in nodes)
        self.label: str = label if label is not None else self.name
        self.show_value: bool = show_value
        self.color: str = color

        # Live operating values updated by the solver after each solve/step.
        self.voltage: float = 0.0
        self.current: float = 0.0
        self.power: float = 0.0

        # Maximum continuous operating ratings used by parameter dialog and stress checks (F1 / F4).
        self.max_power: float = 0.25
        self.max_voltage: float = 50.0
        self.max_current: float = 0.5

    def get_max_ratings(self) -> dict[str, float]:
        """Return max ratings `{max_power, max_voltage, max_current}` for stress checks."""
        return {
            "max_power": float(self.max_power),
            "max_voltage": float(self.max_voltage),
            "max_current": float(self.max_current),
        }

    def set_max_ratings(
        self,
        *,
        max_power: float | None = None,
        max_voltage: float | None = None,
        max_current: float | None = None,
    ) -> None:
        """Update positive continuous rating limits."""
        if max_power is not None:
            if max_power <= 0.0:
                raise ValueError("max_power must be > 0 W.")
            self.max_power = float(max_power)
        if max_voltage is not None:
            if max_voltage <= 0.0:
                raise ValueError("max_voltage must be > 0 V.")
            self.max_voltage = float(max_voltage)
        if max_current is not None:
            if max_current <= 0.0:
                raise ValueError("max_current must be > 0 A.")
            self.max_current = float(max_current)

    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return a short canonical type identifier (e.g., 'resistor', 'dc_voltage')."""

    def get_editable_parameters(self) -> tuple[ComponentParameter, ...]:
        """Return the tuple of editable `ComponentParameter` descriptors for this component."""
        return _DEFAULT_PARAMETERS_BY_TYPE.get(self.component_type, ())

    def get_parameter_value(self, key: str) -> float:
        """Return the current numeric value for parameter `key`."""
        val = getattr(self, key)
        return 0.0 if val is None else float(val)

    def format_parameter_value(self, key: str) -> str:
        """Return the current value of parameter `key` formatted in engineering notation (no unit)."""
        val = getattr(self, key)
        if val is None:
            return "0"
        return format_engineering_value(float(val), unit="")

    def set_parameter_value(self, key: str, raw_value: float | str) -> float:
        """Validate and set parameter `key` using the case-sensitive engineering parser."""
        params = {p.key: p for p in self.get_editable_parameters()}
        if key not in params:
            raise KeyError(
                f"Component {self.name!r} ({self.component_type}) has no editable parameter {key!r}."
            )
        spec = params[key]
        try:
            parsed = parse_engineering_value(raw_value)
        except ValueError as exc:
            raise ValueError(f"{spec.label}: {exc}") from exc

        if not spec.allow_negative and not spec.allow_zero and parsed <= 0.0:
            raise ValueError(
                f"{spec.label} must be > 0 {spec.unit} (got {raw_value!r})."
            )
        if not spec.allow_negative and spec.allow_zero and parsed < 0.0:
            raise ValueError(
                f"{spec.label} cannot be negative (got {raw_value!r})."
            )

        setattr(self, key, parsed)
        return parsed

    def reset_state(self) -> None:
        """Reset transient or thermal state back to initial conditions."""
        self.voltage = 0.0
        self.current = 0.0
        self.power = 0.0

"""Mixed-signal MCU block with 10 MOhm ADC inputs, 25 Ohm digital outputs, and Python callback."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value

MCU_PIN_NAMES: tuple[str, ...] = ("A0", "A1", "D0", "D1", "TX", "RX")
MCU_ANALOG_PINS: tuple[str, ...] = ("A0", "A1")
MCU_DIGITAL_PINS: tuple[str, ...] = ("D0", "D1", "TX", "RX")


class MCUBlock(Component):
    """Programmable mixed-signal microcontroller block for MNA co-simulation.

    Pin order matches `mcu.json` (`nodes = (A0, A1, D0, D1, TX, RX)`):
        - ``A0``, ``A1``: 10 MOhm high-impedance analog ADC inputs to ground.
        - ``D0``, ``D1``, ``TX``: 25 Ohm Thevenin digital outputs (default OUTPUT,
          drive voltage switches smoothly between ``v_low`` and ``v_cc``).
        - ``RX``: 10 MOhm high-impedance digital/UART input to ground (default INPUT).

    Every digital pin may dynamically switch between ``"OUTPUT"`` and ``"INPUT"`` via
    `set_pin_mode(pin, mode)`. Firmware runs purely offline via a local Python
    ``callback(mcu, sim_time, dt)`` on each causal transient sample interval.
    """

    DEFAULT_R_ADC_IN: float = 1e7   # 10 MOhm analog ADC input impedance
    DEFAULT_R_DIG_OUT: float = 25.0  # 25 Ohm CMOS GPIO Thevenin output impedance
    DEFAULT_R_DIG_IN: float = 1e7   # 10 MOhm digital input impedance

    def __init__(
        self,
        name: str,
        node_a0: str | None = None,
        node_a1: str | None = None,
        node_d0: str | None = None,
        node_d1: str | None = None,
        node_tx: str | None = None,
        node_rx: str | None = None,
        *,
        v_cc: float | str = 5.0,
        v_low: float | str = 0.0,
        v_ih_frac: float = 0.6,
        v_il_frac: float = 0.3,
        r_adc_in: float | str = DEFAULT_R_ADC_IN,
        r_dig_out: float | str = DEFAULT_R_DIG_OUT,
        r_dig_in: float | str = DEFAULT_R_DIG_IN,
        adc_bits: int = 10,
        update_period: float | str = 0.0,
        callback: Callable[[MCUBlock, float, float], Mapping[str, bool | float] | None] | None = None,
        initial_outputs: Mapping[str, bool | float] | None = None,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        resolved = (
            node_a0 or f"{name}_A0",
            node_a1 or f"{name}_A1",
            node_d0 or f"{name}_D0",
            node_d1 or f"{name}_D1",
            node_tx or f"{name}_TX",
            node_rx or f"{name}_RX",
        )
        super().__init__(
            name=name,
            nodes=resolved,
            label=label,
            show_value=show_value,
            color=color,
        )
        self.v_cc: float = parse_engineering_value(v_cc)
        self.v_low: float = parse_engineering_value(v_low)
        self.r_adc_in: float = parse_engineering_value(r_adc_in)
        self.r_dig_out: float = parse_engineering_value(r_dig_out)
        self.r_dig_in: float = parse_engineering_value(r_dig_in)
        self.adc_bits: int = max(4, min(24, int(adc_bits)))
        self.update_period: float = max(0.0, parse_engineering_value(update_period))
        self.v_ih_frac: float = float(v_ih_frac)
        self.v_il_frac: float = float(v_il_frac)

        if self.r_adc_in <= 0.0 or self.r_dig_out <= 0.0 or self.r_dig_in <= 0.0:
            raise ValueError(f"MCUBlock {name!r}: pin resistances must be > 0.")

        self.callback = callback
        self._pin_modes: dict[str, str] = {
            "A0": "ANALOG",
            "A1": "ANALOG",
            "D0": "OUTPUT",
            "D1": "OUTPUT",
            "TX": "OUTPUT",
            "RX": "INPUT",
        }
        self._initial_outputs: dict[str, float] = {}
        if initial_outputs:
            for p, val in initial_outputs.items():
                self._initial_outputs[p] = self._coerce_drive_voltage(val)

        self._pin_voltages: dict[str, float] = {p: 0.0 for p in MCU_PIN_NAMES}
        self._output_drives: dict[str, float] = {
            p: self._initial_outputs.get(p, self.v_low) for p in MCU_DIGITAL_PINS
        }
        self._digital_latched: dict[str, bool] = {p: False for p in MCU_PIN_NAMES}
        self._next_eval_time: float = 0.0
        self.user_state: dict[str, Any] = {}

    @property
    def component_type(self) -> str:
        return "mcu"

    @property
    def display_value(self) -> str:
        return f"MCU {self.v_cc:g}V"

    @property
    def pin_names(self) -> tuple[str, ...]:
        return MCU_PIN_NAMES

    def pin_node_map(self) -> dict[str, str]:
        return {pin: node for pin, node in zip(MCU_PIN_NAMES, self.nodes)}

    def _coerce_drive_voltage(self, val: bool | int | float) -> float:
        if isinstance(val, bool):
            return self.v_cc if val else self.v_low
        return float(val)

    def set_pin_mode(self, pin: str, mode: str) -> None:
        if pin not in MCU_DIGITAL_PINS:
            raise KeyError(f"MCUBlock {self.name!r}: digital pin mode invalid for {pin!r}.")
        m = mode.strip().upper()
        if m not in ("INPUT", "OUTPUT"):
            raise ValueError(f"Mode must be 'INPUT' or 'OUTPUT', got {mode!r}.")
        self._pin_modes[pin] = m

    def pin_conductance(self, pin: str) -> float:
        mode = self._pin_modes[pin]
        if mode == "ANALOG":
            return 1.0 / self.r_adc_in
        if mode == "OUTPUT":
            return 1.0 / self.r_dig_out
        return 1.0 / self.r_dig_in

    def is_output_pin(self, pin: str) -> bool:
        return self._pin_modes.get(pin) == "OUTPUT"

    def drive_voltage(self, pin: str) -> float:
        return self._output_drives.get(pin, self.v_low)

    def read_voltage(self, pin: str = "A0") -> float:
        """Return latest measured external node voltage on `pin` (Volts)."""
        if pin not in MCU_PIN_NAMES:
            raise KeyError(f"MCUBlock {self.name!r}: unknown pin {pin!r}.")
        return float(self._pin_voltages.get(pin, 0.0))

    def read_adc(
        self, pin: str = "A0", *, v_ref: float | None = None, bits: int | None = None
    ) -> int:
        """Sample analog voltage on `pin` (`A0` or `A1`) and return quantized ADC code."""
        v = self.read_voltage(pin)
        ref = float(v_ref) if v_ref is not None else self.v_cc
        n_bits = int(bits) if bits is not None else self.adc_bits
        if ref <= 0.0:
            return 0
        max_code = (1 << n_bits) - 1
        norm = min(1.0, max(0.0, v / ref))
        return int(round(norm * max_code))

    def digital_read(self, pin: str) -> bool:
        """Read Schmitt-filtered boolean logic state from any pin."""
        v = self.read_voltage(pin)
        v_il = self.v_low + self.v_il_frac * (self.v_cc - self.v_low)
        v_ih = self.v_low + self.v_ih_frac * (self.v_cc - self.v_low)
        if v >= v_ih:
            self._digital_latched[pin] = True
        elif v <= v_il:
            self._digital_latched[pin] = False
        return self._digital_latched[pin]

    def digital_write(self, pin: str, level: bool | float) -> None:
        """Set digital output drive on `D0`, `D1`, or `TX`/`RX` (when configured OUTPUT)."""
        if pin not in MCU_DIGITAL_PINS:
            raise KeyError(f"MCUBlock {self.name!r}: cannot write to analog pin {pin!r}.")
        self._output_drives[pin] = self._coerce_drive_voltage(level)

    def sense_nodes(self, node_voltages: Mapping[str, float]) -> None:
        """Latch external MNA node voltages onto local pin registers."""
        for pin, node in zip(MCU_PIN_NAMES, self.nodes):
            self._pin_voltages[pin] = float(node_voltages.get(node, 0.0))

    def evaluate_firmware(
        self,
        node_voltages: Mapping[str, float],
        sim_time: float,
        dt: float,
    ) -> bool:
        """Execute offline Python firmware callback if due; return True if any drive changed."""
        self.sense_nodes(node_voltages)
        if self.callback is None:
            return False
        if dt > 0.0 and self.update_period > 0.0 and sim_time + 1e-15 < self._next_eval_time:
            return False

        prev_drives = dict(self._output_drives)
        out_map = self.callback(self, sim_time, dt)
        if isinstance(out_map, Mapping):
            for pin, val in out_map.items():
                if pin in MCU_DIGITAL_PINS:
                    self._output_drives[pin] = self._coerce_drive_voltage(val)

        if dt > 0.0 and self.update_period > 0.0:
            while self._next_eval_time <= sim_time + 1e-15:
                self._next_eval_time += self.update_period

        return any(
            abs(self._output_drives[p] - prev_drives[p]) > 1e-12
            for p in MCU_DIGITAL_PINS
        )

    def reset_state(self) -> None:
        super().reset_state()
        self._pin_voltages = {p: 0.0 for p in MCU_PIN_NAMES}
        self._output_drives = {
            p: self._initial_outputs.get(p, self.v_low) for p in MCU_DIGITAL_PINS
        }
        self._digital_latched = {p: False for p in MCU_PIN_NAMES}
        self._next_eval_time = 0.0
        self.user_state.clear()

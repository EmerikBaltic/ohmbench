"""Sequential Digital Logic & 7-Segment Display models for mixed-signal MNA (no Qt)."""

from __future__ import annotations

from typing import Any, Mapping

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value

_SEVEN_SEG_TABLE: dict[int, tuple[str, ...]] = {
    0x0: ("a", "b", "c", "d", "e", "f"),
    0x1: ("b", "c"),
    0x2: ("a", "b", "d", "e", "g"),
    0x3: ("a", "b", "c", "d", "g"),
    0x4: ("b", "c", "f", "g"),
    0x5: ("a", "c", "d", "f", "g"),
    0x6: ("a", "c", "d", "e", "f", "g"),
    0x7: ("a", "b", "c"),
    0x8: ("a", "b", "c", "d", "e", "f", "g"),
    0x9: ("a", "b", "c", "d", "f", "g"),
    0xA: ("a", "b", "c", "e", "f", "g"),
    0xB: ("c", "d", "e", "f", "g"),
    0xC: ("a", "d", "e", "f"),
    0xD: ("b", "c", "d", "e", "g"),
    0xE: ("a", "d", "e", "f", "g"),
    0xF: ("a", "e", "f", "g"),
}


class DigitalSequentialBlock(Component):
    """Shared CMOS sequential digital macromodel base class (10 MOhm inputs, 25 Ohm outputs)."""

    PIN_NAMES: tuple[str, ...] = ()
    OUTPUT_PINS: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        nodes: tuple[str, ...],
        *,
        v_high: float | str = 5.0,
        v_low: float | str = 0.0,
        v_threshold: float | str = 2.5,
        r_in: float | str = 1e7,
        r_out: float | str = 25.0,
        label: str | None = None,
        show_value: bool = True,
    ) -> None:
        super().__init__(name=name, nodes=nodes, label=label, show_value=show_value)
        self.v_high: float = parse_engineering_value(v_high)
        self.v_low: float = parse_engineering_value(v_low)
        self.v_threshold: float = parse_engineering_value(v_threshold)
        self.r_in: float = parse_engineering_value(r_in)
        self.r_out: float = parse_engineering_value(r_out)
        if self.r_in <= 0.0 or self.r_out <= 0.0:
            raise ValueError(f"{name!r}: r_in and r_out must be > 0 ohm.")
        self._output_drives: dict[str, float] = {p: self.v_low for p in self.OUTPUT_PINS}
        self._prev_clk: float = self.v_low
        self.set_max_ratings(max_power=0.25, max_voltage=7.0, max_current=0.1)

    @property
    def pin_names(self) -> tuple[str, ...]:
        return self.PIN_NAMES

    def is_output_pin(self, pin: str) -> bool:
        return pin in self.OUTPUT_PINS

    def pin_conductance(self, pin: str) -> float:
        return (1.0 / self.r_out) if pin in self.OUTPUT_PINS else (1.0 / self.r_in)

    def drive_voltage(self, pin: str) -> float:
        return self._output_drives.get(pin, self.v_low)

    def _node_for_pin(self, pin: str) -> str:
        return self.nodes[self.PIN_NAMES.index(pin)]

    def _read_bit(self, node_voltages: Mapping[str, float], pin: str) -> bool:
        return float(node_voltages.get(self._node_for_pin(pin), 0.0)) >= self.v_threshold

    def _update_drive(self, pin: str, is_high: bool) -> bool:
        target = self.v_high if is_high else self.v_low
        prev = self._output_drives.get(pin, self.v_low)
        self._output_drives[pin] = target
        return abs(target - prev) > 1e-9

    def reset_state(self) -> None:
        super().reset_state()
        self._prev_clk = self.v_low

    def step_sequential(self, node_voltages: Mapping[str, float]) -> bool:
        """Advance edge-triggered/decoded logic from solved voltages; return True if drives changed."""
        return False


class DFlipFlop(DigitalSequentialBlock):
    """Positive-edge-triggered D Flip-Flop (`D`, `CLK`, `RST`, `Q`, `Q_BAR`)."""

    PIN_NAMES = ("D", "CLK", "RST", "Q", "Q_BAR")
    OUTPUT_PINS = ("Q", "Q_BAR")

    def __init__(
        self,
        name: str,
        node_d: str | None = None,
        node_clk: str | None = None,
        node_rst: str | None = None,
        node_q: str | None = None,
        node_q_bar: str | None = None,
        *,
        initial_q: bool = False,
        **kwargs: Any,
    ) -> None:
        nodes = (
            node_d or f"{name}_D",
            node_clk or f"{name}_CLK",
            node_rst or f"{name}_RST",
            node_q or f"{name}_Q",
            node_q_bar or f"{name}_Q_BAR",
        )
        super().__init__(name, nodes, **kwargs)
        self.initial_q: bool = bool(initial_q)
        self.state_q: bool = bool(initial_q)
        self._sync_outputs()

    @property
    def component_type(self) -> str:
        return "d_flipflop"

    @property
    def display_value(self) -> str:
        return f"Q={'1' if self.state_q else '0'}"

    def _sync_outputs(self) -> bool:
        c1 = self._update_drive("Q", self.state_q)
        c2 = self._update_drive("Q_BAR", not self.state_q)
        return c1 or c2

    def reset_state(self) -> None:
        super().reset_state()
        self.state_q = self.initial_q
        self._sync_outputs()

    def step_sequential(self, node_voltages: Mapping[str, float]) -> bool:
        rst_hi = self._read_bit(node_voltages, "RST")
        curr_clk = float(node_voltages.get(self._node_for_pin("CLK"), 0.0))
        rising = (self._prev_clk < self.v_threshold) and (curr_clk >= self.v_threshold)
        self._prev_clk = curr_clk

        if rst_hi:
            self.state_q = False
        elif rising:
            self.state_q = self._read_bit(node_voltages, "D")
        return self._sync_outputs()

    def to_dict(self) -> dict[str, Any]:
        return {
            "v_high": self.v_high,
            "v_low": self.v_low,
            "v_threshold": self.v_threshold,
            "r_in": self.r_in,
            "r_out": self.r_out,
            "initial_q": self.initial_q,
        }


class JKFlipFlop(DigitalSequentialBlock):
    """Positive-edge-triggered JK Flip-Flop (`J`, `K`, `CLK`, `RST`, `Q`, `Q_BAR`)."""

    PIN_NAMES = ("J", "K", "CLK", "RST", "Q", "Q_BAR")
    OUTPUT_PINS = ("Q", "Q_BAR")

    def __init__(
        self,
        name: str,
        node_j: str | None = None,
        node_k: str | None = None,
        node_clk: str | None = None,
        node_rst: str | None = None,
        node_q: str | None = None,
        node_q_bar: str | None = None,
        *,
        initial_q: bool = False,
        **kwargs: Any,
    ) -> None:
        nodes = (
            node_j or f"{name}_J",
            node_k or f"{name}_K",
            node_clk or f"{name}_CLK",
            node_rst or f"{name}_RST",
            node_q or f"{name}_Q",
            node_q_bar or f"{name}_Q_BAR",
        )
        super().__init__(name, nodes, **kwargs)
        self.initial_q: bool = bool(initial_q)
        self.state_q: bool = bool(initial_q)
        self._sync_outputs()

    @property
    def component_type(self) -> str:
        return "jk_flipflop"

    @property
    def display_value(self) -> str:
        return f"Q={'1' if self.state_q else '0'}"

    def _sync_outputs(self) -> bool:
        c1 = self._update_drive("Q", self.state_q)
        c2 = self._update_drive("Q_BAR", not self.state_q)
        return c1 or c2

    def reset_state(self) -> None:
        super().reset_state()
        self.state_q = self.initial_q
        self._sync_outputs()

    def step_sequential(self, node_voltages: Mapping[str, float]) -> bool:
        rst_hi = self._read_bit(node_voltages, "RST")
        curr_clk = float(node_voltages.get(self._node_for_pin("CLK"), 0.0))
        rising = (self._prev_clk < self.v_threshold) and (curr_clk >= self.v_threshold)
        self._prev_clk = curr_clk

        if rst_hi:
            self.state_q = False
        elif rising:
            j_hi = self._read_bit(node_voltages, "J")
            k_hi = self._read_bit(node_voltages, "K")
            if j_hi and not k_hi:
                self.state_q = True
            elif not j_hi and k_hi:
                self.state_q = False
            elif j_hi and k_hi:
                self.state_q = not self.state_q
        return self._sync_outputs()

    def to_dict(self) -> dict[str, Any]:
        return {
            "v_high": self.v_high,
            "v_low": self.v_low,
            "v_threshold": self.v_threshold,
            "r_in": self.r_in,
            "r_out": self.r_out,
            "initial_q": self.initial_q,
        }


class Counter4Bit(DigitalSequentialBlock):
    """4-Bit Synchronous Binary Ripple Counter (`CLK`, `RST`, `Q0`, `Q1`, `Q2`, `Q3`)."""

    PIN_NAMES = ("CLK", "RST", "Q0", "Q1", "Q2", "Q3")
    OUTPUT_PINS = ("Q0", "Q1", "Q2", "Q3")

    def __init__(
        self,
        name: str,
        node_clk: str | None = None,
        node_rst: str | None = None,
        node_q0: str | None = None,
        node_q1: str | None = None,
        node_q2: str | None = None,
        node_q3: str | None = None,
        *,
        initial_count: int = 0,
        **kwargs: Any,
    ) -> None:
        nodes = (
            node_clk or f"{name}_CLK",
            node_rst or f"{name}_RST",
            node_q0 or f"{name}_Q0",
            node_q1 or f"{name}_Q1",
            node_q2 or f"{name}_Q2",
            node_q3 or f"{name}_Q3",
        )
        super().__init__(name, nodes, **kwargs)
        self.initial_count: int = int(initial_count) & 0xF
        self.count: int = self.initial_count
        self._sync_outputs()

    @property
    def component_type(self) -> str:
        return "counter_4bit"

    @property
    def display_value(self) -> str:
        return f"0x{self.count:X} ({self.count:04b})"

    def _sync_outputs(self) -> bool:
        changed = False
        for bit_idx, pin in enumerate(self.OUTPUT_PINS):
            if self._update_drive(pin, bool((self.count >> bit_idx) & 1)):
                changed = True
        return changed

    def reset_state(self) -> None:
        super().reset_state()
        self.count = self.initial_count
        self._sync_outputs()

    def step_sequential(self, node_voltages: Mapping[str, float]) -> bool:
        rst_hi = self._read_bit(node_voltages, "RST")
        curr_clk = float(node_voltages.get(self._node_for_pin("CLK"), 0.0))
        rising = (self._prev_clk < self.v_threshold) and (curr_clk >= self.v_threshold)
        self._prev_clk = curr_clk

        if rst_hi:
            self.count = 0
        elif rising:
            self.count = (self.count + 1) & 0xF
        return self._sync_outputs()

    def to_dict(self) -> dict[str, Any]:
        return {
            "v_high": self.v_high,
            "v_low": self.v_low,
            "v_threshold": self.v_threshold,
            "r_in": self.r_in,
            "r_out": self.r_out,
            "initial_count": self.initial_count,
        }


class SevenSegDisplay(DigitalSequentialBlock):
    """4-Bit BCD/Hex 7-Segment LED Display (`D0`..`D3` inputs, segments `a..g`)."""

    PIN_NAMES = ("D0", "D1", "D2", "D3")
    OUTPUT_PINS = ()

    def __init__(
        self,
        name: str,
        node_d0: str | None = None,
        node_d1: str | None = None,
        node_d2: str | None = None,
        node_d3: str | None = None,
        **kwargs: Any,
    ) -> None:
        nodes = (
            node_d0 or f"{name}_D0",
            node_d1 or f"{name}_D1",
            node_d2 or f"{name}_D2",
            node_d3 or f"{name}_D3",
        )
        kwargs.setdefault("v_threshold", 2.0)
        super().__init__(name, nodes, **kwargs)
        self.value_nibble: int = 0
        self.decoded_hex_char: str = "0"
        self.active_segments: tuple[str, ...] = _SEVEN_SEG_TABLE[0]

    @property
    def component_type(self) -> str:
        return "seven_seg_display"

    @property
    def display_value(self) -> str:
        return f"HEX '{self.decoded_hex_char}'"

    def reset_state(self) -> None:
        super().reset_state()
        self.value_nibble = 0
        self.decoded_hex_char = "0"
        self.active_segments = _SEVEN_SEG_TABLE[0]

    def step_sequential(self, node_voltages: Mapping[str, float]) -> bool:
        prev_nibble = self.value_nibble
        nibble = 0
        for bit_idx, pin in enumerate(self.PIN_NAMES):
            if self._read_bit(node_voltages, pin):
                nibble |= 1 << bit_idx
        self.value_nibble = nibble
        self.decoded_hex_char = f"{nibble:X}"
        self.active_segments = _SEVEN_SEG_TABLE[nibble]
        return nibble != prev_nibble

    def to_dict(self) -> dict[str, Any]:
        return {
            "v_high": self.v_high,
            "v_low": self.v_low,
            "v_threshold": self.v_threshold,
            "r_in": self.r_in,
        }
